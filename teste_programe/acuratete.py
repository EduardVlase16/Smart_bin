import random
import time
import os
from ultralytics import YOLO

def adauga_zgomot_canal(matrice, probabilitate=0.05):
    """
    Primeste o matrice (lista de liste) si aplica zgomot Salt & Pepper.
    Returneaza:
        - matricea_zgomotoasa
        - masca (1 unde a pus zgomot, 0 unde e curat)
    """
    rows = len(matrice)
    cols = len(matrice[0])
    
    # Copie manuala (deep copy)
    noisy = [r[:] for r in matrice]
    mask = [[0] * cols for _ in range(rows)]
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            val = random.random()
            if val < probabilitate / 2:
                noisy[r][c] = 0        # Pepper
                mask[r][c] = 1
                count += 1
            elif val > 1 - (probabilitate / 2):
                noisy[r][c] = 255      # Salt
                mask[r][c] = 1
                count += 1
                
    return noisy, mask, count

def _calculeaza_acuratete_canal(original, filtrat, masca):
    """Functie interna pentru calculul procentual pe un singur canal."""
    rows = len(original)
    cols = len(original[0])
    
    total_zgomot = 0
    recuperat = 0
    
    for r in range(rows):
        for c in range(cols):
            if masca[r][c] == 1:
                total_zgomot += 1
                # Verificam daca filtrul a adus valoarea aproape de original
                # Toleranta +/- 5 nivele de gri
                if abs(filtrat[r][c] - original[r][c]) <= 5:
                    recuperat += 1
                    
    if total_zgomot == 0:
        return 0.0
    
    return (recuperat / total_zgomot) * 100.0

def ruleaza_test_acuratete(nume_fisier, set_original, set_zgomotos, set_filtrat, set_masti, timp_executie):
    """
    Aceasta este functia pe care o apelezi in MAIN.
    Primeste listele de canale (R, G, B) pentru fiecare etapa.
    """
    # Despachetam canalele
    r_orig, g_orig, b_orig = set_original
    r_no, g_no, b_no = set_zgomotos # Doar informativ daca vrem sa extindem
    r_filt, g_filt, b_filt = set_filtrat
    mask_r, mask_g, mask_b = set_masti
    
    # Calculam acuratetea pe fiecare canal
    acc_r = _calculeaza_acuratete_canal(r_orig, r_filt, mask_r)
    acc_g = _calculeaza_acuratete_canal(g_orig, g_filt, mask_g)
    acc_b = _calculeaza_acuratete_canal(b_orig, b_filt, mask_b)
    
    avg_acc = (acc_r + acc_g + acc_b) / 3
    
    # Generam textul raportului
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log = (
        f"[{timestamp}] TEST NOU\n"
        f"--------------------------------------------------\n"
        f"Acuratete recuperare R: {acc_r:.2f}%\n"
        f"Acuratete recuperare G: {acc_g:.2f}%\n"
        f"Acuratete recuperare B: {acc_b:.2f}%\n"
        f" >> MEDIE TOTALA: {avg_acc:.2f}%\n"
        f"Timp executie filtrare: {timp_executie:.4f} sec\n"
        f"--------------------------------------------------\n"
    )
    
    # Scriem in fisier
    with open(nume_fisier, "a") as f:
        f.write(log)
        
    print(f"\n[INFO] Raport generat in {nume_fisier}")
    print(f"Eficiență medie detectată: {avg_acc:.2f}%")


# ==========================================
#   EXTENSIE PENTRU EVALUARE YOLO
# ==========================================

def evalueaza_acuratete_yolo(folder_test, model_path='weights/best.pt', log_file="rezultate.txt"):
    """
    Ruleaza modelul YOLO pe toate imaginile dintr-un folder si compara
    clasa prezisa cu numele fisierului.
    
    Presupuneri:
    - Numele fisierului contine clasa (ex: 'Cardboard_49.jpg')
    - Numele fisierului poate avea separatori '_' sau '-'
    """
    
    if not os.path.exists(folder_test):
        print(f"[EROARE] Folderul {folder_test} nu exista!")
        return
    
    if not os.path.exists(model_path):
        print(f"[EROARE] Modelul {model_path} nu a fost gasit!")
        # Fallback pe model standard daca nu exista cel custom
        model_path = 'yolo11n.pt' 
        print(" -> Se incearca modelul standard yolo11n.pt...")

    print(f"\n[YOLO TEST] Se incarca modelul: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[EROARE] Nu s-a putut incarca modelul: {e}")
        return

    # Extragem lista de imagini
    fisiere = [f for f in os.listdir(folder_test) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_imagini = len(fisiere)
    
    if total_imagini == 0:
        print("[YOLO TEST] Nu s-au gasit imagini in folder.")
        return

    corecte = 0
    gresite = 0
    nedetectate = 0
    
    print(f"[YOLO TEST] Se proceseaza {total_imagini} imagini...")
    start_time = time.time()

    for nume_img in fisiere:
        cale_completa = os.path.join(folder_test, nume_img)
        
        # Rulăm predicția (verbose=False ca sa nu umple consola)
        results = model(cale_completa, verbose=False, conf=0.25)
        
        # Verificăm dacă a detectat ceva
        if len(results[0].boxes) > 0:
            # Luăm clasa cu cea mai mare confidență (prima din listă)
            cls_id = int(results[0].boxes.cls[0])
            nume_clasa_predisa = model.names[cls_id].lower()
            
            # Normalizam numele fisierului pentru comparatie
            # Ex: "Food-Organics_01.jpg" -> "food organics 01.jpg"
            nume_fisier_clean = nume_img.lower().replace('-', ' ').replace('_', ' ')
            
            # Verificam daca predictia exista in numele fisierului
            # Ex: daca "plastic" se gaseste in "plastic 893.jpg"
            if nume_clasa_predisa in nume_fisier_clean:
                corecte += 1
            else:
                gresite += 1
                # Optional: Uncomment pentru a vedea erorile in consola
                # print(f"GRESIT: {nume_img} | Predictie: {nume_clasa_predisa}")
        else:
            nedetectate += 1
            # print(f"NEDETECTAT: {nume_img}")

    end_time = time.time()
    durata = end_time - start_time
    acuratete = (corecte / total_imagini) * 100.0

    # Generare Raport Text
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_yolo = (
        f"[{timestamp}] TEST ACURATETE YOLO\n"
        f"--------------------------------------------------\n"
        f"Model utilizat: {model_path}\n"
        f"Folder testat: {folder_test}\n"
        f"Total imagini: {total_imagini}\n"
        f" >> CORECTE: {corecte}\n"
        f" >> GRESITE: {gresite}\n"
        f" >> FARA DETECTIE: {nedetectate}\n"
        f"EFICIENTA MODEL: {acuratete:.2f}%\n"
        f"Timp executie: {durata:.2f} sec\n"
        f"--------------------------------------------------\n\n"
    )

    # Scriere in fisier (Append)
    with open(log_file, "a") as f:
        f.write(log_yolo)

    print(f"\n[REZULTAT] Eficiență YOLO: {acuratete:.2f}%")
    print(f"Raportul a fost adaugat in '{log_file}'.")

def ruleaza_test_comparativ(nume_fisier, set_original, set_masti, 
                            set_fuzzy, timp_fuzzy, 
                            set_median, timp_median):
    """
    Compara eficienta Fuzzy vs Median si scrie raportul.
    """
    # Despachetam seturile
    r_orig, g_orig, b_orig = set_original
    mask_r, mask_g, mask_b = set_masti
    
    r_fuz, g_fuz, b_fuz = set_fuzzy
    r_med, g_med, b_med = set_median
    
    # --- Calculam Acuratete FUZZY ---
    acc_r_fuz = _calculeaza_acuratete_canal(r_orig, r_fuz, mask_r)
    acc_g_fuz = _calculeaza_acuratete_canal(g_orig, g_fuz, mask_g)
    acc_b_fuz = _calculeaza_acuratete_canal(b_orig, b_fuz, mask_b)
    avg_fuz = (acc_r_fuz + acc_g_fuz + acc_b_fuz) / 3
    
    # --- Calculam Acuratete MEDIAN ---
    acc_r_med = _calculeaza_acuratete_canal(r_orig, r_med, mask_r)
    acc_g_med = _calculeaza_acuratete_canal(g_orig, g_med, mask_g)
    acc_b_med = _calculeaza_acuratete_canal(b_orig, b_med, mask_b)
    avg_med = (acc_r_med + acc_g_med + acc_b_med) / 3
    
    # Generam textul raportului
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log = (
        f"[{timestamp}] TEST COMPARATIV (Fuzzy vs Median)\n"
        f"--------------------------------------------------\n"
        f"METODA      | R (%)  | G (%)  | B (%)  | MEDIA  | TIMP (s)\n"
        f"--------------------------------------------------\n"
        f"FUZZY (Tu)  | {acc_r_fuz:5.2f} | {acc_g_fuz:5.2f} | {acc_b_fuz:5.2f} | {avg_fuz:5.2f}% | {timp_fuzzy:.4f}\n"
        f"MEDIAN (Std)| {acc_r_med:5.2f} | {acc_g_med:5.2f} | {acc_b_med:5.2f} | {avg_med:5.2f}% | {timp_median:.4f}\n"
        f"--------------------------------------------------\n"
        f"Diferenta (Fuzzy - Median): {avg_fuz - avg_med:.2f}%\n"
        f"--------------------------------------------------\n"
    )
    
    # Scriem in fisier
    with open(nume_fisier, "a") as f:
        f.write(log)
        
    print(f"\n[RAPORT] Fuzzy: {avg_fuz:.2f}% vs Median: {avg_med:.2f}%")
    print(f"Detalii salvate in {nume_fisier}")