import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os


img_path = r'C:\Users\alexandru.vlase\Desktop\Proiecte\Imagini test\imagine segmentare otsu multiprag.jpg'

def plot_histogram(roi_gray, ax):
    """
    Funcție ajutătoare pentru a desena histograma pixelilor.
    Aceasta arată distribuția intensității luminii (matematica din spatele threshold-ului).
    """
    # Calculăm histograma
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
    ax.plot(hist, color='black')
    ax.set_title('Histograma ROI (Distribuția intensității)')
    ax.set_xlabel('Intensitate Pixel (0-255)')
    ax.set_ylabel('Număr Pixeli')
    ax.grid(True, alpha=0.3)

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("image_path", type=str, help="Calea către imagine")
    # parser.add_argument("--conf", type=float, default=0.25, help="Prag încredere")
    # args = parser.parse_args()

    # Definim manual parametrii pe care îi lua args înainte
    class Args:
        pass
    args = Args()
    args.image_path = img_path  # Aici folosim variabila ta de sus
    args.conf = 0.25            # Setăm manual confidența

    # 1. Încărcare Model și Imagine
    model_path = 'weights/best.pt'
    if not os.path.exists(model_path):
        print("Modelul nu a fost găsit.")
        return
        
    model = YOLO(model_path)
    original_image = cv2.imread(args.image_path)
    
    # Facem o copie pentru contur manual
    contour_image = original_image.copy()
    
    # Convertim BGR la RGB pentru Matplotlib (OpenCV foloseste BGR)
    img_rgb_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 2. Detecție
    results = model(args.image_path, conf=args.conf)
    
    # Imaginea 1: Rezultatul standard YOLO
    yolo_plot = results[0].plot() 
    yolo_plot_rgb = cv2.cvtColor(yolo_plot, cv2.COLOR_BGR2RGB)

    # Pregătim figura Matplotlib
    # Vom avea 2 rânduri și 2 coloane
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Analiza Detecție Gunoi - {args.image_path}', fontsize=16)

    # --- CADRAN 1: Detecția Standard YOLO ---
    axs[0, 0].imshow(yolo_plot_rgb)
    axs[0, 0].set_title("1. Detecție Standard (YOLO Bounding Box)")
    axs[0, 0].axis('off')

    # Variabile pentru a stoca datele despre primul obiect detectat (pentru grafice)
    roi_gray_sample = None
    binary_mask_sample = None
    
    # 3. Procesare Manuală Contur
    if results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Decupare ROI (Region of Interest)
            roi = contour_image[y1:y2, x1:x2]
            
            # --- Procesare Imagine (Matematica aplicată) ---
            # A. Grayscale (Reducere dimensionalitate)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # B. Gaussian Blur (Reducere Zgomot - Convoluție)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # C. Thresholding (Binarizare Otsu - Minimizare varianță)
            # Folosim THRESH_BINARY_INV presupunând că gunoiul e mai închis la culoare? 
            # Daca gunoiul e deschis si fundalul inchis, scoate _INV.
            # Otsu calculează automat pragul optim.
            val_prag, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Salvăm datele pentru grafice (doar pentru primul obiect)
            if roi_gray_sample is None:
                roi_gray_sample = gray
                binary_mask_sample = binary

            # D. Găsire Contururi (Topologie Suzuki)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # E. Desenare Contur
            # Ajustăm coordonatele înapoi la imaginea mare
            for cnt in contours:
                cnt[:, :, 0] += x1
                cnt[:, :, 1] += y1
                cv2.drawContours(contour_image, [cnt], -1, (0, 255, 0), 2)
            
            # Adăugăm eticheta text (fără cutie)
            cls_name = model.names[int(box.cls[0])]
            cv2.putText(contour_image, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- CADRAN 2: Contur Manual ---
    contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
    axs[0, 1].imshow(contour_image_rgb)
    axs[0, 1].set_title("2. Contur Calculat Manual (OpenCV)")
    axs[0, 1].axis('off')

    # --- CADRAN 3: Diagrama de Zgomot (Histograma) ---
    if roi_gray_sample is not None:
        plot_histogram(roi_gray_sample, axs[1, 0])
        axs[1, 0].axvline(x=val_prag, color='r', linestyle='--', label=f'Prag Otsu: {val_prag}')
        axs[1, 0].legend()
        axs[1, 0].set_title("3. Analiza Matematică (Histogramă ROI)\nCe e la stânga liniei roșii e fundal, la dreapta e obiect")
    else:
        axs[1, 0].text(0.5, 0.5, "Nu s-au detectat obiecte", ha='center')

    # --- CADRAN 4: Masca Binară (Rezultatul Binarizării) ---
    if binary_mask_sample is not None:
        axs[1, 1].imshow(binary_mask_sample, cmap='gray')
        axs[1, 1].set_title("4. Masca Binară (Ce 'vede' algoritmul de contur)")
        axs[1, 1].axis('off')
    else:
        axs[1, 1].text(0.5, 0.5, "Nu s-au detectat obiecte", ha='center')

    # Afișare finală
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()