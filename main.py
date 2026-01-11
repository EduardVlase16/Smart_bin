import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import detector.librarie_ica as lib
import teste_programe.acuratete as acc
import detector.detect_image as rn


#img_path = r'C:\Users\alexandru.vlase\Desktop\Proiecte\Python eu\Smart_Bin_SIB\data\archive\realwaste-main\RealWaste\Plastic\Plastic_893.jpg'
#img_path = r'C:\Users\alexandru.vlase\Desktop\Proiecte\Python eu\Smart_Bin_SIB\data\archive\realwaste-main\RealWaste\Cardboard\Cardboard_440.jpg'
#img_path = r'C:\Users\alexandru.vlase\Desktop\Proiecte\Python eu\Smart_Bin_SIB\data\archive\realwaste-main\RealWaste\Metal\Metal_763.jpg'
#img_path = r'C:\Users\alexandru.vlase\Desktop\Proiecte\Python eu\Smart_Bin_SIB\data\archive\realwaste-main\RealWaste\Food Organics\Food Organics_366.jpg'
img_path = r'C:\Users\alexandru.vlase\Desktop\Proiecte\Imagini test\imagine segmentare otsu multiprag.jpg'

otsu_config = {
    'k_classes' : 3,
    'step' : 2
}

#  DEFINIREA MULTIMILOR FUZZY (INPUT)

# Intrare 1: "Diferenta" absoluta fata de vecini (|Pixel - Media|)
# Categorii: Mica (Zgomot mic/Nimic), Mare (Posibil Zgomot sau Muchie)
# Format [x_points], [u_points] (membership values)
fuzzy_config = {
    'x_diff' : [
    [0, 10, 20],       # Mica: Scade de la 1 la 0 intre 0 si 20
    [10, 30, 255]      # Mare: Creste de la 0 la 1 intre 10 si 30
    ],
    'u_diff' : [
    [1, 1, 0],         # Membership pt "Mica"
    [0, 1, 1]          # Membership pt "Mare"
    ],
    'x_context' : [
    [0, 10, 30],       # Neted: Zona uniforma
    [15, 40, 255]      # Muchie: Zona cu detalii
    ],
    'u_context' : [
    [1, 1, 0],         # Membership pt "Neted"
    [0, 1, 1]          # Membership pt "Muchie"
    ],
    'n_context' : 2,

# DEFINIREA IESIRILOR (SINGLETONS - SUGENO 0-order)
# Iesire: Factorul de corectie Alpha (0 = Pastreaza original, 1 = Fa media)
# Valori: [Nu_Corecta, Corecteaza_Total]
    'y_vals' : [0.0, 1.0],

# BAZA DE REGULI (Rule Base)
# Matrice [Idx_Diff][Idx_Context] => Indexul Iesirii y_vals
# Regula 1: Daca Dif=Mica -> Corectie=Nu (indiferent de context)
# Regula 2: Daca Dif=Mare SI Context=Muchie -> Corectie=Nu (nu strica detaliile!)
# Regula 3: Daca Dif=Mare SI Context=Neted -> Corectie=Da (e zgomot pe fundal!)
    'rules' : [
    [0, 0],  # Diff=Mica, Context=Neted -> 0 (Nu corecta)
    [1, 0],  # Diff=Mare, Context=Neted-> 0 (Corecteaza)
    ]
}


# ==========================================
#  MAIN RGB
# ==========================================

try:
    img = mpimg.imread(img_path)
    # Convertim la 0-255 int daca e float
    if img.max() <= 1.0:
        img = (img * 255).astype(int)
    
    # Eliminam canalul Alpha daca exista (RGBA -> RGB)
    if len(img[0][0]) == 4:
         img = [row[:, :3] for row in img] # Simplificare, sau slice manual

    print(f"Imagine incarcata. Procesare RGB cu K={otsu_config['k_classes']} nivele/canal...")

     # A. Aplicam Zgomot (folosind noul modul)
    PROCENT_ZGOMOT = 0.08 # 8%
    
    # 1. Separam canalele
    r_raw, g_raw, b_raw = lib.split_channels(img)
    lib.scrie_valori_ordonate(r_raw, g_raw, b_raw, "rezultate.txt")
    start_time = time.time()

    # 2. Aplicam Otsu pentru fiecare canal independent
    # Canal ROSU
    hist_r, tot_r = lib.compute_histogram(r_raw)
    th_r = lib.multi_otsu_solver(hist_r, tot_r, otsu_config)
    print(f"Praguri Rosu: {th_r}")
    r_seg = lib.apply_channel_segmentation(r_raw, th_r)
    r_noisy, mask_r, _ = acc.adauga_zgomot_canal(r_seg, PROCENT_ZGOMOT)
    smoothed_r = lib.apply_fuzzy_smoothing(r_noisy, fuzzy_config)

    # Canal VERDE
    hist_g, tot_g = lib.compute_histogram(g_raw)
    th_g = lib.multi_otsu_solver(hist_g, tot_g, otsu_config)
    print(f"Praguri Verde: {th_g}")
    g_seg = lib.apply_channel_segmentation(g_raw, th_g)
    g_noisy, mask_g, _ = acc.adauga_zgomot_canal(g_seg, PROCENT_ZGOMOT)
    smoothed_g = lib.apply_fuzzy_smoothing(g_noisy, fuzzy_config)

    # Canal ALBASTRU
    hist_b, tot_b = lib.compute_histogram(b_raw)
    th_b = lib.multi_otsu_solver(hist_b, tot_b, otsu_config)
    print(f"Praguri Albastru: {th_b}")
    b_seg = lib.apply_channel_segmentation(b_raw, th_b)
    b_noisy, mask_b, _ = acc.adauga_zgomot_canal(b_seg, PROCENT_ZGOMOT)
    smoothed_b = lib.apply_fuzzy_smoothing(b_noisy, fuzzy_config)

    end_time = time.time()
    durata = end_time - start_time
    print(f"Timp total procesare: {durata:.2f} secunde")

    # C. Validare si Scriere in Fisier (O singura linie)
    acc.ruleaza_test_acuratete(
        "rezultate.txt",
        set_original=(r_seg, g_seg, b_seg),
        set_zgomotos=(r_noisy, g_noisy, b_noisy),
        set_filtrat=(smoothed_r, smoothed_g, smoothed_b),
        set_masti=(mask_r, mask_g, mask_b),
        timp_executie=durata
    )

    # 3. Recombinam canalele
    final_img = lib.merge_channels(r_seg, g_seg, b_seg)
    noisy_img = lib.merge_channels(r_noisy, g_noisy, b_noisy)
    fuzzy_img = lib.merge_channels(smoothed_r, smoothed_g, smoothed_b)

    # Definim calea unde salvam imaginea temporara
    final_img_path = "rezultat_segmentare_temp.jpg"
    
    # Salvam imaginea din memorie pe disc
    # .astype(np.uint8) asigura ca pixelii sunt intre 0-255 (format standard imagine)
    plt.imsave(final_img_path, np.array(final_img).astype(np.uint8))
    
    print(f"Imaginea segmentata a fost salvata temporar la: {final_img_path}")

    # 4. Afisare
    # --- FIGURA 1: Original vs Segmentat ---
    plt.figure(figsize=(12, 6))  # O fereastra noua de dimensiuni 12x6
    plt.suptitle("Etapa 1: Segmentare Otsu") # Titlul ferestrei

    # Subplot 1 din 2 (Stanga)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original RGB")
    plt.axis('off')

    # Subplot 2 din 2 (Dreapta)
    plt.subplot(1, 2, 2)
    plt.imshow(final_img)
    plt.title(f"Segmentat RGB\n({otsu_config['k_classes']}^3 = {otsu_config['k_classes']**3} culori posibile)")
    plt.axis('off')

    # --- FIGURA 2: Zgomot vs Filtrare ---
    plt.figure(figsize=(12, 6)) # A doua fereastra noua
    plt.suptitle("Etapa 2: Eliminare Zgomot (Fuzzy)")

    # Subplot 1 din 2 (Stanga)
    plt.subplot(1, 2, 1)
    plt.imshow(noisy_img)
    plt.title("Imagine Zgomotoasa (Intrare Filtru)")
    plt.axis('off')

    # Subplot 2 din 2 (Dreapta)
    plt.subplot(1, 2, 2)
    plt.imshow(fuzzy_img)
    plt.title("Rezultat Final (Filtrata Fuzzy)")
    plt.axis('off')

    # Figura 3 si 4, punem rezultatele retelei neuronale
    rn.detectie_imagine(img_path)
    rn.detectie_imagine(final_img_path)

    plt.show()

except FileNotFoundError:
    print("Nu am gasit imaginea. Verifica calea.")
except Exception as e:
    print(f"Eroare: {e}")