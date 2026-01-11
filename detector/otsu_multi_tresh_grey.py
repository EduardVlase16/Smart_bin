import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

# ==========================================
# CONFIGURARE
# ==========================================
# Schimba calea catre imaginea ta aici
image_path = r'C:\Users\alexandru.vlase\Desktop\Proiecte\Imagini test\imagine segmentare otsu multiprag.jpg' 

# Numarul de clase (K=5 inseamna 4 praguri de taiere)
K_CLASSES = 4 

# Pasul de cautare (Mariti la 4-8 pentru viteza, 1 pentru precizie maxima dar lent)
SEARCH_STEP = 2 

# ==========================================
# 1. FUNCTII UTILITARE (Fara librarii de procesare)
# ==========================================

def manual_rgb_to_gray(img_data):
    """
    Converteste o imagine RGB in Grayscale folosind formula standard:
    Y = 0.299*R + 0.587*G + 0.114*B
    """
    rows = len(img_data)
    cols = len(img_data[0])
    
    # Verificam daca e deja grayscale
    if not isinstance(img_data[0][0], (list, tuple)) and len(img_data.shape) == 2:
        return img_data

    gray_img = []
    for r in range(rows):
        row_pixels = []
        for c in range(cols):
            pixel = img_data[r][c]
            # Extragem R, G, B (unele formate au si Alpha, il ignoram)
            r_val, g_val, b_val = pixel[0], pixel[1], pixel[2]
            
            # Formula de luminanta
            gray_val = int(0.299 * r_val + 0.587 * g_val + 0.114 * b_val)
            row_pixels.append(gray_val)
        gray_img.append(row_pixels)
    return gray_img

def compute_histogram(gray_img):
    """ Numara aparitia fiecarei intensitati (0-255) """
    hist = [0] * 256
    total_pixels = 0
    for row in gray_img:
        for val in row:
            hist[val] += 1
            total_pixels += 1
    return hist, total_pixels

# ==========================================
# 2. ALGORITMUL OTSU MULTI-NIVEL (Core)
# ==========================================

def get_variance_for_region(start, end, P, S, total_mean):
    """
    Calculeaza contributia la varianta inter-clasa pentru un interval dat [start, end].
    Formula: weight * (class_mean - total_mean)^2
    """
    if start > end: return 0.0

    # Folosim tabelele cumulative pentru viteza O(1)
    # P[x] este suma probabilitatilor de la 0 la x
    w = P[end] - (P[start-1] if start > 0 else 0)
    
    if w == 0: return 0.0 # Clasa goala

    # S[x] este suma (i * p_i) de la 0 la x
    sum_val = S[end] - (S[start-1] if start > 0 else 0)
    
    mu = sum_val / w
    return w * ((mu - total_mean) ** 2)

def recursive_otsu_search(start_idx, k_remaining, P, S, total_mean, memo):
    """
    Cauta recursiv pragurile care maximizeaza varianta.
    """
    # Cheie pentru memoization (cache)
    memo_key = (start_idx, k_remaining)
    if memo_key in memo:
        return memo[memo_key]

    # Caz de baza: Mai avem o singura clasa de alocat (ultima)
    # Ea va lua tot ce a ramas pana la 255.
    if k_remaining == 1:
        var = get_variance_for_region(start_idx, 255, P, S, total_mean)
        return var, []

    max_var = -1.0
    best_threshs = []

    # Cautam unde sa punem urmatorul prag (t)
    # Mergem pana la 255 - (k_remaining - 1) pentru a lasa loc celorlalte clase
    # Optimizare: folosim SEARCH_STEP pentru a sari peste valori
    for t in range(start_idx, 256 - k_remaining, SEARCH_STEP):
        
        # Varianta clasei curente (de la start_idx la t)
        current_class_var = get_variance_for_region(start_idx, t, P, S, total_mean)
        
        # Recursivitate: Gaseste maximul pentru restul imaginii
        remaining_var, remaining_threshs = recursive_otsu_search(t + 1, k_remaining - 1, P, S, total_mean, memo)
        
        total_var = current_class_var + remaining_var

        if total_var > max_var:
            max_var = total_var
            best_threshs = [t] + remaining_threshs

    memo[memo_key] = (max_var, best_threshs)
    return max_var, best_threshs

def multi_otsu_solver(hist, total_pixels, k_classes):
    """ Functia principala care pregateste datele si porneste recursivitatea """
    
    # 1. Normalizare (Probabilitati)
    probs = [h / total_pixels for h in hist]

    # 2. Construim Tabelele Cumulative (Look-up Tables)
    P = [0.0] * 256
    S = [0.0] * 256
    
    P[0] = probs[0]
    S[0] = 0 * probs[0]
    
    for i in range(1, 256):
        P[i] = P[i-1] + probs[i]
        S[i] = S[i-1] + (i * probs[i])

    total_mean = S[255] # Media globala a imaginii
    
    # 3. Pornim cautarea recursiva
    print(f"Calculare Otsu pentru K={k_classes} clase (step={SEARCH_STEP})... Asteptati...")
    start_time = time.time()
    
    memo = {} # Cache pentru a nu recalcula aceleasi sub-probleme
    final_var, thresholds = recursive_otsu_search(0, k_classes, P, S, total_mean, memo)
    
    end_time = time.time()
    print(f"Gata! Durata: {end_time - start_time:.2f} secunde.")
    print(f"Praguri gasite: {thresholds}")
    
    return thresholds

def apply_segmentation(gray_img, thresholds):
    """ Reconstruieste imaginea folosind pragurile gasite """
    rows = len(gray_img)
    cols = len(gray_img[0])
    
    # Definim culorile (intensitatile) pentru fiecare zona
    # Le impartim echidistant intre 0 si 255
    k = len(thresholds) + 1
    colors = [int(i * 255 / (k - 1)) for i in range(k)]
    
    segmented_img = []
    for r in range(rows):
        new_row = []
        for c in range(cols):
            val = gray_img[r][c]
            
            # Determinam in ce clasa cade pixelul
            class_idx = 0
            for i, th in enumerate(thresholds):
                if val > th:
                    class_idx = i + 1
                else:
                    break
            
            new_row.append(colors[class_idx])
        segmented_img.append(new_row)
    return segmented_img

# ==========================================
# 3. MAIN
# ==========================================

try:
    # A. Incarcare Imagine
    # Folosim matplotlib doar pentru citire (returneaza numpy array, dar il tratam ca lista)
    img = mpimg.imread(image_path)
    
    # Daca imaginea e float (0-1), o convertim la 0-255
    if img.max() <= 1.0:
        img = (img * 255).astype(int)

    # B. Conversie Grayscale Manuala
    print("Conversie la Grayscale...")
    gray_image = manual_rgb_to_gray(img)
    
    # C. Calcul Histograma
    hist, total_pixels = compute_histogram(gray_image)

    # D. Algoritmul Otsu Multi-Nivel
    thresholds = multi_otsu_solver(hist, total_pixels, K_CLASSES)

    # E. Aplicarea rezultatului
    segmented_image = apply_segmentation(gray_image, thresholds)

    # F. Afisare Grafice
    plt.figure(figsize=(12, 6))

    # Plot 1: Imaginea Originala Grayscale
    plt.subplot(1, 3, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title("Original Grayscale")
    plt.axis('off')

    # Plot 2: Histograma si Pragurile
    plt.subplot(1, 3, 2)
    plt.plot(hist, color='black')
    plt.title(f"Histograma si {K_CLASSES-1} Praguri")
    for th in thresholds:
        plt.axvline(x=th, color='red', linestyle='--')
    plt.xlim([0, 255])

    # Plot 3: Imaginea Segmentata
    plt.subplot(1, 3, 3)
    plt.imshow(segmented_image, cmap='gray')
    plt.title(f"Segmentare Otsu (K={K_CLASSES})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Eroare: Nu am gasit imaginea la calea: {image_path}")
    print("Te rog modifica variabila 'image_path' din cod.")
except Exception as e:
    print(f"A aparut o eroare: {e}")