import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import detector.librarie_ica as lib
import teste_programe.acuratete as acc
import detector.detect_image as rn
import os

# CALEA CATRE IMAGINE (Modifica aici daca vrei alta poza)
img_path = r'C:\Users\alexandru.vlase\Desktop\Proiecte\Imagini test\imagine segmentare otsu multiprag.jpg'

# CALEA CATRE DATASET-UL DE TEST PENTRU YOLO
folder_cu_imagini_test = r'C:\Users\alexandru.vlase\Desktop\Proiecte\Python eu\Smart_Bin_SIB\Dataset-labeling-1\test\images'

otsu_config = {
    'k_classes' : 3,
    'step' : 2
}

# ==========================================
#  CONFIGURARE FUZZY
# ==========================================
fuzzy_config = {
    'x_diff' : [
        [0, 10, 20],       # Mica
        [10, 30, 255]      # Mare
    ],
    'u_diff' : [
        [1, 1, 0],         
        [0, 1, 1]          
    ],
    'x_context' : [
        [0, 10, 30],       # Neted
        [15, 40, 255]      # Muchie
    ],
    'u_context' : [
        [1, 1, 0],         
        [0, 1, 1]          
    ],
    'n_context' : 2,
    'y_vals' : [0.0, 1.0], # 0 = Nu corecta, 1 = Corecteaza
    'rules' : [
        [0, 0],  # Diff=Mica -> 0
        [1, 0],  # Diff=Mare, Context=Neted -> 0 (Corecteaza - nota: verifica daca regula e setata cum vrei tu, aici e 0 in codul tau anterior)
    ]
}
# Nota: La rules, [1, 0] inseamna output index 0 (adica y_vals[0] = 0.0). 
# Daca voiai sa corecteze (y=1.0), ar trebui sa fie indexul 1. 
# Verifica logica ta Fuzzy daca vrei sa netezeasca zgomotul (de obicei Diff Mare + Neted => Corectie).

# ==========================================
#  MAIN PROGRAM
# ==========================================

try:
    # 1. Incarcare Imagine
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Imaginea nu exista: {img_path}")

    img = mpimg.imread(img_path)
    
    # Normalizare la 0-255 int
    if img.max() <= 1.0:
        img = (img * 255).astype(int)
    
    # Eliminare canal Alpha
    if len(img.shape) > 2 and img.shape[2] == 4:
         img = img[:, :, :3]

    print(f"Imagine incarcata. Procesare RGB cu K={otsu_config['k_classes']} nivele/canal...")

    PROCENT_ZGOMOT = 0.08 # 8%
    
    # 2. Separare Canale
    r_raw, g_raw, b_raw = lib.split_channels(img)
    lib.scrie_valori_ordonate(r_raw, g_raw, b_raw, "rezultate.txt")
    
    start_time = time.time()

    # 3. Procesare pe fiecare Canal (Otsu -> Zgomot -> Fuzzy & Median)

    # --- CANAL ROSU ---
    print("--- Procesare Canal ROSU ---")
    hist_r, tot_r = lib.compute_histogram(r_raw)
    th_r = lib.multi_otsu_solver(hist_r, tot_r, otsu_config) # Calculam pragurile
    r_seg = lib.apply_channel_segmentation(r_raw, th_r)      # Cream r_seg (Segmentat)
    
    # Adaugam Zgomot
    r_noisy, mask_r, _ = acc.adauga_zgomot_canal(r_seg, PROCENT_ZGOMOT)
    
    # Filtru Fuzzy
    t0 = time.time()
    smoothed_r = lib.apply_fuzzy_smoothing(r_noisy, fuzzy_config)
    t_fuz_r = time.time() - t0
    
    # Filtru Median (Competitor)
    t0 = time.time()
    median_r = lib.apply_median_filter(r_noisy)
    t_med_r = time.time() - t0

    # --- CANAL VERDE ---
    print("--- Procesare Canal VERDE ---")
    hist_g, tot_g = lib.compute_histogram(g_raw)
    th_g = lib.multi_otsu_solver(hist_g, tot_g, otsu_config)
    g_seg = lib.apply_channel_segmentation(g_raw, th_g)      # Cream g_seg
    
    g_noisy, mask_g, _ = acc.adauga_zgomot_canal(g_seg, PROCENT_ZGOMOT)
    
    t0 = time.time()
    smoothed_g = lib.apply_fuzzy_smoothing(g_noisy, fuzzy_config)
    t_fuz_g = time.time() - t0
    
    t0 = time.time()
    median_g = lib.apply_median_filter(g_noisy)
    t_med_g = time.time() - t0

    # --- CANAL ALBASTRU ---
    print("--- Procesare Canal ALBASTRU ---")
    hist_b, tot_b = lib.compute_histogram(b_raw)
    th_b = lib.multi_otsu_solver(hist_b, tot_b, otsu_config)
    b_seg = lib.apply_channel_segmentation(b_raw, th_b)      # Cream b_seg
    
    b_noisy, mask_b, _ = acc.adauga_zgomot_canal(b_seg, PROCENT_ZGOMOT)
    
    t0 = time.time()
    smoothed_b = lib.apply_fuzzy_smoothing(b_noisy, fuzzy_config)
    t_fuz_b = time.time() - t0
    
    t0 = time.time()
    median_b = lib.apply_median_filter(b_noisy)
    t_med_b = time.time() - t0

    # Calculam timpii totali
    total_time_fuzzy = t_fuz_r + t_fuz_g + t_fuz_b
    total_time_median = t_med_r + t_med_g + t_med_b
    
    print(f"Gata procesarea imaginilor.")

    # 4. Validare si Scriere Raport Comparativ
    acc.ruleaza_test_comparativ(
        "rezultate.txt",
        set_original=(r_seg, g_seg, b_seg),     # Comparam cu imaginea segmentata curata
        set_masti=(mask_r, mask_g, mask_b),
        set_fuzzy=(smoothed_r, smoothed_g, smoothed_b),
        timp_fuzzy=total_time_fuzzy,
        set_median=(median_r, median_g, median_b),
        timp_median=total_time_median
    )

    # 5. Recombinare Canale pentru Afisare
    final_img = lib.merge_channels(r_seg, g_seg, b_seg)
    noisy_img = lib.merge_channels(r_noisy, g_noisy, b_noisy)
    fuzzy_img = lib.merge_channels(smoothed_r, smoothed_g, smoothed_b)
    median_img = lib.merge_channels(median_r, median_g, median_b)

    # Salvam temporar pentru YOLO
    final_img_path = "rezultat_segmentare_temp.jpg"
    plt.imsave(final_img_path, np.array(final_img).astype(np.uint8))

    # 6. Afisare Grafica
    
    # Fereastra 1: Original vs Segmentat
    plt.figure(figsize=(10, 5))
    plt.suptitle("Etapa 1: Segmentare Otsu")
    plt.subplot(1, 2, 1); plt.imshow(img); plt.title("Original"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(final_img); plt.title("Segmentat Otsu"); plt.axis('off')

    # Fereastra 2: Comparatie Filtre (Zgomot vs Fuzzy vs Median)
    plt.figure(figsize=(15, 6))
    plt.suptitle("Etapa 2: Eliminare Zgomot (Comparatie)")
    
    plt.subplot(1, 3, 1)
    plt.imshow(noisy_img)
    plt.title("Imagine cu Zgomot (Input)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(fuzzy_img)
    plt.title(f"Filtru FUZZY\nTimp: {total_time_fuzzy:.3f}s")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(median_img)
    plt.title(f"Filtru MEDIAN\nTimp: {total_time_median:.3f}s")
    plt.axis('off')

    # 7. Detectie YOLO pe o singura imagine (demo)
    print("\n--- Rulare YOLO Demo ---")
    rn.detectie_imagine(img_path)       # Pe original
    rn.detectie_imagine(final_img_path) # Pe segmentat

    # 8. Test Acuratete YOLO pe tot folderul
    print("\n--- Start Test Acuratete YOLO (Folder) ---")
    acc.evalueaza_acuratete_yolo(
        folder_test=folder_cu_imagini_test,
        model_path='weights/best.pt', 
        log_file="rezultate.txt"
    )

    plt.show()

except FileNotFoundError as e:
    print(f"EROARE FISIER: {e}")
except Exception as e:
    print(f"EROARE GENERALA: {e}")
    import traceback
    traceback.print_exc()