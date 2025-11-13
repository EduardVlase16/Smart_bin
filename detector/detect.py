import cv2
from ultralytics import YOLO
import math

# Încarcă modelul tău antrenat
# Asigură-te că fișierul 'best.pt' se află în folderul 'weights'
try:
    model = YOLO('../weights/best.pt')
except Exception as e:
    print(f"Eroare la încărcarea modelului: {e}")
    print("Asigură-te că ai descărcat modelul 'best.pt' și l-ai pus în folderul 'weights/'.")
    exit()

# --- IMPORTANT! ---
# Actualizează această listă cu numele claselor tale
# EXACT așa cum le-ai definit în Roboflow (ordinea contează).
# Poți verifica ordinea în fișierul 'data.yaml' din exportul Roboflow.

# --- MODIFICARE ---
# Am actualizat această listă bazat pe log-ul tău de antrenare din Colab.
# VERIFICĂ DACĂ ORDINEA ESTE CORECTĂ (trebuie să se potrivească cu 'data.yaml' din Colab)
class_names = ["Cardboard", "Metal", "Plastic", "Waste"] # <-- ACTUALIZAT AICI

# Încearcă să deschizi camera web
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Eroare: Nu se poate deschide camera web.")
    exit()

cap.set(3, 640) # Setează lățimea
cap.set(4, 480) # Setează înălțimea

print("Camera s-a deschis. Apasă 'q' pentru a închide.")

while True:
    success, img = cap.read()
    if not success:
        print("Eroare: Nu se poate citi frame-ul de la cameră.")
        break

    # Rulează detecția YOLOv8 pe frame
    # stream=True este mai eficient pentru video
    results = model(img, stream=True)

    # Parcurge rezultatele detecției
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Extrage coordonatele casetei
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Desenează caseta
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Extrage încrederea (confidence)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            
            # Extrage clasa
            cls_index = int(box.cls[0])
            
            # Verifică dacă indexul clasei este valid
            if cls_index < len(class_names):
                cls_name = class_names[cls_index]
                label = f'{cls_name}: {confidence}'
                
                # Calculează dimensiunea textului pentru a desena un fundal
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Desenează fundalul textului
                cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), (255, 0, 255), -1)
                
                # Scrie textul (numele clasei și încrederea)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                print(f"Atenție: Indexul clasei {cls_index} este în afara limitelor listei 'class_names'.")


    # Afișează imaginea rezultată
    cv2.imshow('SmartBin Detector (Apasă q pentru a ieși)', img)

    # Oprește bucla dacă se apasă tasta 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Eliberează resursele
cap.release()
cv2.destroyAllWindows()
print("Camera s-a închis.")