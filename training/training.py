from ultralytics import YOLO
import torch

def train_local():
    # Verifică dacă ai GPU (CUDA) disponibil
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Se antrenează pe dispozitivul: {device}")
    
    # Încarcă un model de bază (de ex. yolov8n.pt)
    model = YOLO('yolov8n.pt')

    # Antrenează modelul
    # NOTĂ: Trebuie să descarci setul de date de pe Roboflow (format YOLOv8)
    # și să pui aici calea corectă către fișierul 'data.yaml'
    # De exemplu: 'data/Gunoi-1/data.yaml'
    try:
        results = model.train(
            data='CALEA/CATRE/data.yaml',  # <-- ACTUALIZEAZĂ ACEASTĂ CALE
            epochs=50,
            imgsz=640,
            device=device
        )
        print("Antrenare finalizată cu succes!")
        print("Modelul tău 'best.pt' se află în folderul 'runs/detect/train/weights/'")
        print("Copiază-l manual în folderul 'weights/' al proiectului.")
    except Exception as e:
        print(f"A apărut o eroare la antrenare: {e}")
        print("Verifică dacă ai specificat corect calea către 'data.yaml'.")

if __name__ == '__main__':
    print("ATENȚIE: Antrenarea locală este foarte lentă fără un GPU NVIDIA (CUDA).")
    print("Este recomandat să folosești Google Colab (vezi training_colab.py).")
    train_local()