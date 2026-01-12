import argparse
import cv2
import os
import matplotlib.pyplot as plt # Importam matplotlib
from ultralytics import YOLO


def detectie_imagine(path):
    # 1. Configurare Argumente
    img_path = path

    class Args:
        pass
    args = Args()
    args.image_path = img_path 
    args.conf = 0.25           

    # 2. Verificări preliminare
    if not os.path.exists(args.image_path):
        print(f"Eroare: Imaginea '{args.image_path}' nu a fost găsită!")
        return

    model_path = 'weights/best.pt'
    # model_path = 'yolo11n.pt' # Uncomment daca nu ai weights custom
    
    if not os.path.exists(model_path):
        print(f"Eroare: Modelul '{model_path}' nu a fost găsit!")
        return

    # 3. Încărcare Model
    print(f"Se încarcă modelul din {model_path}...")
    model = YOLO(model_path)

    # 4. Predicție
    print(f"Se procesează imaginea: {args.image_path}...")
    results = model(args.image_path, conf=args.conf)

    # 5. Procesare Rezultate pentru Matplotlib
    # .plot() returnează un numpy array în format BGR
    annotated_frame_bgr = results[0].plot()

    # --- PAS CRITIC: Conversie BGR -> RGB ---
    # Matplotlib vrea RGB, OpenCV ne da BGR. Fara asta, culorile sunt gresite.
    annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)

    # 6. Afișare cu Matplotlib (Fereastra 3)
    plt.figure(figsize=(10, 8)) # Dimensiune fereastra
    plt.suptitle("Detectie Obiecte (YOLO)")
    
    plt.imshow(annotated_frame_rgb)
    plt.axis('off') # Ascundem axele cu numere
