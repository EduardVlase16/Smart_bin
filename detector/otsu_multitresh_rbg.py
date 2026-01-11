import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

# ==========================================
# CONFIGURARE
# ==========================================
image_path = r'C:\Users\alexandru.vlase\Desktop\Smart_Bin_SIB\data\archive\realwaste-main\RealWaste\Cardboard\Cardboard_54.jpg' 

# Numarul de clase PER CANAL.
# Daca K_PER_CHANNEL = 2, vei avea 2x2x2 = 8 culori in final.
# Daca K_PER_CHANNEL = 4, vei avea 4x4x4 = 64 culori.
K_PER_CHANNEL = 6 

# Pasul de cautare (mai mare = mai rapid)
SEARCH_STEP = 1

# ==========================================
# 1. FUNCTII UTILITARE RGB
# ==========================================

def scrie_valori_ordonate(r_ch, g_ch, b_ch, nume_fisier="valori_canale.txt"):
    """
    Preia cele 3 canale (matrici 2D), extrage valorile unice, 
    le sorteaza si le scrie in formatul cerut.
    """
    
    # Functie interna (helper) pentru a procesa un singur canal
    # Aceasta evita repetarea codului de 3 ori
    def proceseaza_canal(matrice_canal):
        valori_unice = set() # Folosim un SET pentru a elimina automat duplicatele
        
        for rand in matrice_canal:
            for pixel in rand:
                # Convertim la int pentru a scapa de formatul np.uint8 daca exista
                valori_unice.add(int(pixel))
        
        # Sortam valorile de la mic la mare
        # sorted() returneaza o lista
        return sorted(list(valori_unice))

    # 1. Procesam datele
    print("Se extrag si sorteaza valorile unice pentru R, G, B...")
    r_sorted = proceseaza_canal(r_ch)
    g_sorted = proceseaza_canal(g_ch)
    b_sorted = proceseaza_canal(b_ch)

    # 2. Scriem in fisier
    with open(nume_fisier, "w") as f:
        # Scriem Canalul Rosu
        # Convertim lista de numere intr-un singur sir de text separate prin virgula
        str_rosu = ", ".join(map(str, r_sorted))
        f.write(f"canal Rosu: {str_rosu}\n")
        
        # Scriem Canalul Verde
        str_verde = ", ".join(map(str, g_sorted))
        f.write(f"canal Verde: {str_verde}\n")
        
        # Scriem Canalul Albastru
        str_albastru = ", ".join(map(str, b_sorted))
        f.write(f"canal Albastru: {str_albastru}\n")

    print(f"Gata! Valorile au fost scrise in '{nume_fisier}'.")

def split_channels(img_data):
    """ Separa imaginea in liste R, G, B separate """
    rows = len(img_data)
    cols = len(img_data[0])
    
    r_ch, g_ch, b_ch = [], [], []
    
    for r in range(rows):
        row_r, row_g, row_b = [], [], []
        for c in range(cols):
            pixel = img_data[r][c]
            row_r.append(pixel[0])
            row_g.append(pixel[1])
            row_b.append(pixel[2])
        r_ch.append(row_r)
        g_ch.append(row_g)
        b_ch.append(row_b)

    return r_ch, g_ch, b_ch

def merge_channels(r_ch, g_ch, b_ch):
    """ Combina canalele inapoi intr-o imagine RGB """
    rows = len(r_ch)
    cols = len(r_ch[0])
    new_img = []
    
    for r in range(rows):
        new_row = []
        for c in range(cols):
            # Reconstituim pixelul [R, G, B]
            new_row.append([r_ch[r][c], g_ch[r][c], b_ch[r][c]])
        new_img.append(new_row)
    return new_img

def compute_histogram(channel_data):
    """ Histograma pentru un singur canal """
    hist = [0] * 256
    total_pixels = 0
    for row in channel_data:
        for val in row:
            hist[val] += 1
            total_pixels += 1
    return hist, total_pixels

# ==========================================
# 2. ALGORITMUL OTSU MULTI-NIVEL (Neschimbat)
# ==========================================
# Aceasta parte ramane identica matematic, deoarece matematica
# variatiei este aceeasi indiferent daca numerele reprezinta R, G sau B.

def get_variance_for_region(start, end, P, S, total_mean):
    if start > end: return 0.0
    w = P[end] - (P[start-1] if start > 0 else 0)
    if w == 0: return 0.0
    sum_val = S[end] - (S[start-1] if start > 0 else 0)
    mu = sum_val / w
    return w * ((mu - total_mean) ** 2)

def recursive_otsu_search(start_idx, k_remaining, P, S, total_mean, memo):
    memo_key = (start_idx, k_remaining)
    if memo_key in memo: return memo[memo_key]

    if k_remaining == 1:
        var = get_variance_for_region(start_idx, 255, P, S, total_mean)
        return var, []

    max_var = -1.0
    best_threshs = []

    # Cautare optimizata cu SEARCH_STEP
    for t in range(start_idx, 256 - k_remaining, SEARCH_STEP):
        current_class_var = get_variance_for_region(start_idx, t, P, S, total_mean)
        remaining_var, remaining_threshs = recursive_otsu_search(t + 1, k_remaining - 1, P, S, total_mean, memo)
        total_var = current_class_var + remaining_var

        if total_var > max_var:
            max_var = total_var
            best_threshs = [t] + remaining_threshs

    memo[memo_key] = (max_var, best_threshs)
    return max_var, best_threshs

def multi_otsu_solver(hist, total_pixels, k_classes):
    probs = [h / total_pixels for h in hist]
    P = [0.0] * 256
    S = [0.0] * 256
    P[0] = probs[0]
    S[0] = 0.0
    for i in range(1, 256):
        P[i] = P[i-1] + probs[i]
        S[i] = S[i-1] + (i * probs[i])
    total_mean = S[255]
    
    memo = {}
    _, thresholds = recursive_otsu_search(0, k_classes, P, S, total_mean, memo)
    return thresholds

def apply_channel_segmentation(channel_data, thresholds):
    """ Aplica pragurile pe un singur canal """
    rows = len(channel_data)
    cols = len(channel_data[0])
    
    # Calculam valorile de inlocuire (centrul intervalelor)
    # Ex: pt 2 clase (1 prag), valorile vor fi aprox 0 si 255 (sau media zonei)
    # Aici folosim o impartire echidistanta simplificata pentru vizualizare
    k = len(thresholds) + 1
    values = [int(i * 255 / (k - 1)) for i in range(k)]
    
    new_channel = []
    for r in range(rows):
        new_row = []
        for c in range(cols):
            val = channel_data[r][c]
            class_idx = 0
            for i, th in enumerate(thresholds):
                if val > th:
                    class_idx = i + 1
                else:
                    break
            new_row.append(values[class_idx])
        new_channel.append(new_row)
    return new_channel

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

def apply_fuzzy_smoothing(colour_ch):
    rows = len(colour_ch)
    cols = len(colour_ch[0])
    fuzzy_img = [[0 for _ in range(cols)] for _ in range(rows)]
    
    print("Se aplica filtrarea Fuzzy CORECTATA...")
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            
            # 1. Colectam vecinii (inclusiv centrul initial)
            neighbors = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    neighbors.append(colour_ch[r+i][c+j])
            
            center_pixel = colour_ch[r][c]
            
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
            fuzzy_img[r][c] = int(new_val)
            
    return fuzzy_img

# ==========================================
# 3. MAIN RGB
# ==========================================

try:
    img = mpimg.imread(image_path)
    # Convertim la 0-255 int daca e float
    if img.max() <= 1.0:
        img = (img * 255).astype(int)
    
    # Eliminam canalul Alpha daca exista (RGBA -> RGB)
    if len(img[0][0]) == 4:
         img = [row[:, :3] for row in img] # Simplificare, sau slice manual

    print(f"Imagine incarcata. Procesare RGB cu K={K_PER_CHANNEL} nivele/canal...")
    
    # 1. Separam canalele
    r_raw, g_raw, b_raw = split_channels(img)
    scrie_valori_ordonate(r_raw, g_raw, b_raw, "rezultate.txt")
    start_time = time.time()

    # 2. Aplicam Otsu pentru fiecare canal independent
    # Canal ROSU
    hist_r, tot_r = compute_histogram(r_raw)
    th_r = multi_otsu_solver(hist_r, tot_r, K_PER_CHANNEL)
    print(f"Praguri Rosu: {th_r}")
    r_seg = apply_channel_segmentation(r_raw, th_r)
    smoothed_r = apply_fuzzy_smoothing(r_seg)

    # Canal VERDE
    hist_g, tot_g = compute_histogram(g_raw)
    th_g = multi_otsu_solver(hist_g, tot_g, K_PER_CHANNEL)
    print(f"Praguri Verde: {th_g}")
    g_seg = apply_channel_segmentation(g_raw, th_g)
    smoothed_g = apply_fuzzy_smoothing(g_seg)

    # Canal ALBASTRU
    hist_b, tot_b = compute_histogram(b_raw)
    th_b = multi_otsu_solver(hist_b, tot_b, K_PER_CHANNEL)
    print(f"Praguri Albastru: {th_b}")
    b_seg = apply_channel_segmentation(b_raw, th_b)
    smoothed_b = apply_fuzzy_smoothing(b_seg)

    end_time = time.time()
    print(f"Timp total procesare: {end_time - start_time:.2f} secunde")

    # 3. Recombinam canalele
    final_img = merge_channels(r_seg, g_seg, b_seg)
    fuzzy_img = merge_channels(smoothed_r, smoothed_g, smoothed_b)

    # 4. Afisare
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original RGB")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(final_img)
    plt.title(f"Segmentat RGB\n({K_PER_CHANNEL}^3 = {K_PER_CHANNEL**3} culori posibile)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(fuzzy_img)
    plt.title("Filtrata Fuzzy")
    plt.axis('off')

    plt.show()

except FileNotFoundError:
    print("Nu am gasit imaginea. Verifica calea.")
except Exception as e:
    print(f"Eroare: {e}")