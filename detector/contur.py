import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

image_path = r'C:\Users\alexandru.vlase\Desktop\Proiecte\Imagini test\imagine segmentare otsu multiprag.jpg' 

# ==========================================
# 1. LOGICA FUZZY (Portata din C++)
# ==========================================

# A. DEFINIREA MULTIMILOR FUZZY (INPUT)

# Intrare 1: "Diferenta" absoluta fata de vecini (|Pixel - Media|)
# Categorii: Mica (Zgomot mic/Nimic), Mare (Posibil Zgomot sau Muchie)
# Format [x_points], [u_points] (membership values)
x_diff = [
    [0, 10, 20],       # Mica: Scade de la 1 la 0 intre 0 si 20
    [10, 30, 255]      # Mare: Creste de la 0 la 1 intre 10 si 30
]
u_diff = [
    [1, 1, 0],         # Membership pt "Mica"
    [0, 1, 1]          # Membership pt "Mare"
]
n_diff = 2 # Numar functii apartenenta pt Diferenta

# Intrare 2: "Context" (Variația locală: Max_vecini - Min_vecini)
# Categorii: Neted (Smooth), Muchie (Edge)
x_context = [
    [0, 10, 30],       # Neted: Zona uniforma
    [15, 40, 255]      # Muchie: Zona cu detalii
]
u_context = [
    [1, 1, 0],         # Membership pt "Neted"
    [0, 1, 1]          # Membership pt "Muchie"
]
n_context = 2

# B. DEFINIREA IESIRILOR (SINGLETONS - SUGENO 0-order)
# Iesire: Factorul de corectie Alpha (0 = Pastreaza original, 1 = Fa media)
# Valori: [Nu_Corecta, Corecteaza_Total]
y_vals = [0.0, 1.0] 

# C. BAZA DE REGULI (Rule Base)
# Matrice [Idx_Diff][Idx_Context] => Indexul Iesirii y_vals
# Regula 1: Daca Dif=Mica -> Corectie=Nu (indiferent de context)
# Regula 2: Daca Dif=Mare SI Context=Muchie -> Corectie=Nu (nu strica detaliile!)
# Regula 3: Daca Dif=Mare SI Context=Neted -> Corectie=Da (e zgomot pe fundal!)
rules = [
    [0, 0],  # Diff=Mica, Context=Neted -> 0 (Nu corecta)
    [0, 0],  # Diff=Mica, Context=Muchie-> 0 (Nu corecta)
    # Atentie: Aici e diferenta fata de C++ (am simplificat matricea)
]
# Pentru Diff=Mare:
rules_high_diff = [1, 0] # [Neted->Corecteaza, Muchie->Nu Corecta]


# --- Implementarea Functiei C++ "grad_apart" in Python ---
def grad_apart(x, xA, uA):
    """
    Calculeaza gradul de apartenenta (fuzificarea).
    Portare directa a functiei din exemplul C++.
    """
    nA = len(xA)
    # Cazuri extreme (in afara definitiei)
    if x < xA[0]: return uA[0]
    if x > xA[nA-1]: return uA[nA-1]
    
    for i in range(nA - 1):
        if x >= xA[i] and x <= xA[i+1]:
            # Panta crescatoare
            if uA[i] < uA[i+1]:
                return (x - xA[i]) * (uA[i+1] - uA[i]) / (xA[i+1] - xA[i]) + uA[i]
            # Platou
            if uA[i] == uA[i+1]:
                return uA[i]
            # Panta descrescatoare
            if uA[i] > uA[i+1]:
                return (xA[i+1] - x) * (uA[i] - uA[i+1]) / (xA[i+1] - xA[i]) + uA[i+1]
    return 0.0

# --- Sistemul de Inferenta Sugeno ---
def sugeno_inference(diff_val, context_val):
    """
    Echivalentul functiei Sugeno3 din C++.
    """
    # 1. Fuzificare (Calculam gradele de apartenenta mu)
    mu_diff = []
    for i in range(n_diff):
        mu_diff.append(grad_apart(diff_val, x_diff[i], u_diff[i]))
        
    mu_context = []
    for i in range(n_context):
        mu_context.append(grad_apart(context_val, x_context[i], u_context[i]))
        
    # 2. Evaluarea Regulilor (Inference)
    # Calculam forta de tragere a fiecarei reguli (firing strength)
    # Folosim operatorul MIN pentru AND (ca in C++)
    
    numerator = 0.0
    denominator = 0.0
    
    # Iteram manual prin regulile logice definite mental mai sus
    
    # R1: Diff Mica (indiferent de context) -> Y=0
    # Forta regulii este data doar de mu_diff[0] (Mica)
    w1 = mu_diff[0] 
    z1 = y_vals[0] # 0.0
    numerator += w1 * z1
    denominator += w1
    
    # R2: Diff Mare SI Context Neted -> Y=1 (Corecteaza)
    w2 = min(mu_diff[1], mu_context[0]) 
    z2 = y_vals[1] # 1.0
    numerator += w2 * z2
    denominator += w2
    
    # R3: Diff Mare SI Context Muchie -> Y=0 (Protejeaza muchia)
    w3 = min(mu_diff[1], mu_context[1])
    z3 = y_vals[0] # 0.0
    numerator += w3 * z3
    denominator += w3
    
    # 3. Defuzificare (Weighted Average)
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ==========================================
# 2. PROCESARE IMAGINE (Fara librarii externe)
# ==========================================

def manual_rgb_to_gray(img_data):
    # (Codul tau anterior de conversie)
    rows = len(img_data)
    cols = len(img_data[0])
    gray_img = []
    for r in range(rows):
        row_pixels = []
        for c in range(cols):
            pixel = img_data[r][c]
            r_val, g_val, b_val = pixel[0], pixel[1], pixel[2]
            gray_val = int(0.299 * r_val + 0.587 * g_val + 0.114 * b_val)
            row_pixels.append(gray_val)
        gray_img.append(row_pixels)
    return gray_img

def apply_fuzzy_smoothing(gray_img):
    rows = len(gray_img)
    cols = len(gray_img[0])
    new_img = [[0 for _ in range(cols)] for _ in range(rows)]
    
    print("Se aplica filtrarea Fuzzy CORECTATA...")
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            
            # 1. Colectam vecinii (inclusiv centrul initial)
            neighbors = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    neighbors.append(gray_img[r+i][c+j])
            
            center_pixel = gray_img[r][c]
            
            # 2. SEPARAM CENTRUL DE VECINI
            # Avem 9 elemente. Elementul cu indexul 4 este centrul (r, c)
            neighbors_excl_center = neighbors[:4] + neighbors[5:]
            
            # 3. Calculam statistica DOAR pe vecini (fara centru)
            local_mean = sum(neighbors_excl_center) / 8
            
            # --- MODIFICAREA CRITICA AICI ---
            # Contextul se calculeaza EXCLUSIV pe vecini.
            # Daca centrul e 255 si vecinii 0 -> contextul va fi 0 (Neted), nu 255 (Muchie)
            local_min = min(neighbors_excl_center)
            local_max = max(neighbors_excl_center)
            
            context_val = local_max - local_min
            # --------------------------------
            
            # Diff ramane: cat de diferit e centrul de media vecinilor
            diff_val = abs(center_pixel - local_mean)
            
            # Rulam inferenta
            correction_factor = sugeno_inference(diff_val, context_val)
            
            # Aplicam corectia
            new_val = (1.0 - correction_factor) * center_pixel + correction_factor * local_mean
            new_img[r][c] = int(new_val)
            
    return new_img

# def apply_fuzzy_smoothing(gray_img):
#     rows = len(gray_img)
#     cols = len(gray_img[0])
    
#     # Imaginea noua
#     new_img = [[0 for _ in range(cols)] for _ in range(rows)]
    
#     print("Se aplica filtrarea Fuzzy (poate dura putin)...")
    
#     # Parcurgem imaginea (fara margini pt simplificare 3x3)
#     for r in range(1, rows - 1):
#         for c in range(1, cols - 1):
            
#             # Extragem fereastra 3x3
#             neighbors = []
#             for i in range(-1, 2):
#                 for j in range(-1, 2):
#                     neighbors.append(gray_img[r+i][c+j])
            
#             center_pixel = gray_img[r][c]
            
#             # Calculam statisticile locale
#             # Media vecinilor (fara pixelul central pentru a detecta zgomotul mai bine)
#             neighbors_excl_center = neighbors[:4] + neighbors[5:]
#             local_mean = sum(neighbors_excl_center) / 8
            
#             local_min = min(neighbors)
#             local_max = max(neighbors)
            
#             # --- INPUTURILE PENTRU SISTEMUL FUZZY ---
#             # 1. Diff: Diferenta absoluta fata de medie
#             diff_val = abs(center_pixel - local_mean)
            
#             # 2. Context: Range-ul local (Max - Min)
#             # Daca diferenta e mica, e zona neteda. Daca e mare, e muchie.
#             context_val = local_max - local_min
            
#             # --- RULAM SUGENO ---
#             correction_factor = sugeno_inference(diff_val, context_val)
            
#             # --- APLICAM CORECTIA ---
#             # NewPixel = (1 - alpha) * Old + alpha * Mean
#             new_val = (1.0 - correction_factor) * center_pixel + correction_factor * local_mean
            
#             new_img[r][c] = int(new_val)
            
#     return new_img

# ==========================================
# 3. MAIN
# ==========================================

try:
    img = mpimg.imread(image_path)
    if img.max() <= 1.0: img = (img * 255).astype(int)
    
    # Convertim la Gray
    gray = manual_rgb_to_gray(img)
    
    # Adaugam putin zgomot artificial pentru a testa (optional)
    import random
    rows, cols = len(gray), len(gray[0])
    noisy_gray = [row[:] for row in gray]
    for _ in range(300): # 300 pixeli de zgomot
        r, c = random.randint(1, rows-2), random.randint(1, cols-2)
        noisy_gray[r][c] = 255 if random.random() > 0.5 else 0

    # Aplicam Fuzzy Smoothing
    start = time.time()
    smoothed = apply_fuzzy_smoothing(noisy_gray)
    end = time.time()
    print(f"Gata in {end-start:.2f} secunde.")

    # Afisare
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Originala")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_gray, cmap='gray')
    plt.title("Cu Zgomot (Input)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(smoothed, cmap='gray')
    plt.title("Filtrare Fuzzy Sugeno")
    plt.axis('off')
    
    plt.show()

except FileNotFoundError:
    print("Imaginea nu a fost gasita.")