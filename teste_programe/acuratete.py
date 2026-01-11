import random
import time

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