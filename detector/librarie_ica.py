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
# 2. ALGORITMUL OTSU MULTI-NIVEL
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

def recursive_otsu_search(start_idx, k_remaining, P, S, total_mean, memo, step):
    memo_key = (start_idx, k_remaining)
    if memo_key in memo: return memo[memo_key]

    if k_remaining == 1:
        var = get_variance_for_region(start_idx, 255, P, S, total_mean)
        return var, []

    max_var = -1.0
    best_threshs = []

    # Cautare optimizata cu SEARCH_STEP
    for t in range(start_idx, 256 - k_remaining, step):
        current_class_var = get_variance_for_region(start_idx, t, P, S, total_mean)
        remaining_var, remaining_threshs = recursive_otsu_search(t + 1, k_remaining - 1, P, S, total_mean, memo, step)
        total_var = current_class_var + remaining_var

        if total_var > max_var:
            max_var = total_var
            best_threshs = [t] + remaining_threshs

    memo[memo_key] = (max_var, best_threshs)
    return max_var, best_threshs

def multi_otsu_solver(hist, total_pixels, config):
    k_classes = config.get('k_classes', 3)
    step = config.get('step', 2)
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
    _, thresholds = recursive_otsu_search(0, k_classes, P, S, total_mean, memo, step)
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

# ==========================================
# 3. LOGICA FUZZY PENTRU NETEZIRE MARGINI
# ==========================================
# Aceste functii sunt adaptare proprie la alfgoritmul folosit la concurs, un sistem fuzzy cu 2 intrari
# si o iesire de tip sugeno al carui rol este de a stabili daca pixelul face parte dinn margini sau nu si pentru a netezi contururile

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
def sugeno_inference(diff_val, context_val, config):
    """
    Echivalentul functiei Sugeno3 din C++.
    """
    x_diff = config['x_diff']
    u_diff = config['u_diff']
    x_context = config['x_context']
    u_context = config['u_context']
    y_vals = config['y_vals']
    rules = config.get('rules') 

    n_diff = len(x_diff)
    n_context = len(x_context)

    # 1. Fuzificare (Calculam gradele de apartenenta mu)
    mu_diff = []
    for i in range(n_diff):
        mu_diff.append(grad_apart(diff_val, x_diff[i], u_diff[i]))
        
    mu_context = []
    for i in range(n_context):
        mu_context.append(grad_apart(context_val, x_context[i], u_context[i]))
        
    # Daca avem matricea in config, o folosim dinamic
    numerator = 0.0
    denominator = 0.0
    if rules:
        for i in range(n_diff):          # Itereaza prin starile Diff (Mica, Mare...)
            for j in range(n_context):   # Itereaza prin starile Context (Neted, Muchie...)
                
                # Calculeaza forta regulii (AND = min)
                weight = min(mu_diff[i], mu_context[j])
                
                # Citeste din matrice ce iesire trebuie sa avem (0 sau 1)
                output_idx = rules[i][j]
                y_val = y_vals[output_idx]
                
                numerator += weight * y_val
                denominator += weight
    else:
        # Fallback (Plan B): Logica hardcodata daca uiti sa pui matricea in main
        # R1: Diff Mica -> 0
        w1 = mu_diff[0]
        numerator += w1 * y_vals[0]
        denominator += w1
        
        # R2: Diff Mare + Neted -> 1
        w2 = min(mu_diff[1], mu_context[0])
        numerator += w2 * y_vals[1]
        denominator += w2
        
        # R3: Diff Mare + Muchie -> 0
        w3 = min(mu_diff[1], mu_context[1])
        numerator += w3 * y_vals[0]
        denominator += w3

    # 4. Defuzificare
    if denominator == 0:
        return 0.0
    return numerator / denominator

# ==========================================
# -- PROCESAREA FUZZY -> RETURNEAZA IMAGINEA NETEZITA --
# ==========================================

def apply_fuzzy_smoothing(colour_ch, config):
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
            correction_factor = sugeno_inference(diff_val, context_val, config)
            
            # Aplicam corectia
            new_val = (1.0 - correction_factor) * center_pixel + correction_factor * local_mean
            fuzzy_img[r][c] = int(new_val)
            
    return fuzzy_img

def apply_median_filter(channel_data):
    """
    Aplica filtru median standard 3x3.
    """
    rows = len(channel_data)
    cols = len(channel_data[0])
    
    # Cream matricea rezultat (copie goala)
    filtered_img = [[0 for _ in range(cols)] for _ in range(rows)]
    
    print("Se aplica filtrul Median (competitor)...")
    
    # Evitam marginile (similar cu Fuzzy)
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            
            # 1. Colectam vecinii 3x3
            neighbors = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    neighbors.append(channel_data[r+i][c+j])
            
            # 2. Sortam lista
            neighbors.sort()
            
            # 3. Alegem elementul din mijloc (indexul 4 din 0..8)
            median_val = neighbors[4]
            
            filtered_img[r][c] = median_val
            
    return filtered_img