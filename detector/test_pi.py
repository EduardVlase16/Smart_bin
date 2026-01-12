import time
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2, Picamera2Config

# --- INCARCARE MODEL ---
print("Se incarca modelul...")
try:
    model = YOLO("best.pt")
except:
    print("Nu gasesc best.pt! Folosesc yolov8n.pt standard pentru test.")
    model = YOLO("yolov8n.pt")

class_names = ["Cardboard", "Metal", "Plastic", "Waste"]

# --- SETUP CAMERĂ NATIV (PICAMERA2) ---
print("Configurare Picamera2 (Nativ Pi 5)...")
picam2 = Picamera2()

# Configuram rezolutia si formatul. Pi 5 prefera formatul "main"
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)

# Pornim camera
picam2.start()
print("Camera a pornit! Intram in bucla...")

# --- BUCLA PRINCIPALA ---
try:
    while True:
        # 1. Capturam imaginea direct ca array (mult mai stabil decat cv2)
        # Aceasta functie asteapta urmatorul frame disponibil
        image = picam2.capture_array()

        # Picamera da RGB, OpenCV vrea BGR. Facem conversia.
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 2. Inferenta YOLO
        results = model(frame, stream=True, verbose=False)

        # 3. Procesare rezultate
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Coordonate
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Incredere si Clasa
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Desenare
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Afisare text (cu verificare index)
                if cls_id < len(class_names):
                    label = f"{class_names[cls_id]} {conf:.2f}"
                else:
                    label = f"Obj {cls_id} {conf:.2f}"
                
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. Afisare (Folosind flag-ul pentru a evita erorile Wayland)
        cv2.imshow("Smart Bin Pi5", frame)

        # Iesire cu q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Eroare in timpul rularii: {e}")

finally:
    print("Oprire camera...")
    picam2.stop()
    cv2.destroyAllWindows()