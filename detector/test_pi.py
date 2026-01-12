import time
import sys
import cv2
import numpy as np
# Fortam printarea imediata in terminal (flush=True)
print(">>> [1/6] Importurile au reusit.", flush=True)

from ultralytics import YOLO
from picamera2 import Picamera2

# --- CONFIGURARE ---
print(">>> [2/6] Initializare Picamera2...", flush=True)
try:
    picam2 = Picamera2()
    # Configurare explicita pentru performanta si compatibilitate
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    print(">>> [SUCCESS] Camera a pornit cu succes!", flush=True)
except Exception as e:
    print(f"!!! [EROARE] Nu pot porni camera: {e}", flush=True)
    sys.exit()

# --- INCARCARE MODEL ---
print(">>> [3/6] Incarcare model YOLO (asteapta putin)...", flush=True)
try:
    # Incearca best.pt, daca nu, fallback pe yolov8n.pt
    try:
        model = YOLO("../weights/best.pt") # Ajusteaza calea daca e nevoie
        print(">>> [INFO] Am incarcat 'best.pt'", flush=True)
    except:
        print("!!! [ATENTIE] Nu gasesc best.pt, descarc yolov8n.pt standard...", flush=True)
        model = YOLO("yolov8n.pt")
    
    print(">>> [SUCCESS] Model incarcat!", flush=True)
except Exception as e:
    print(f"!!! [EROARE] Problema la model: {e}", flush=True)
    sys.exit()

class_names = ["Cardboard", "Metal", "Plastic", "Waste"]

print(">>> [4/6] Incep bucla de detectie. Fereastra ar trebui sa apara acum.", flush=True)
print(">>> Apasa 'q' in fereastra video pentru a iesi.", flush=True)

frame_count = 0

try:
    while True:
        # 1. CAPTURA
        # Captureaza imaginea ca array numpy
        image = picam2.capture_array()
        
        # Picamera2 da RGB, OpenCV vrea BGR
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 2. DETECTIE
        results = model(frame, stream=True, verbose=False)

        # 3. DESENARE
        detected_something = False
        for r in results:
            for box in r.boxes:
                detected_something = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Culoare verde
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"Obj {cls_id}"
                if cls_id < len(class_names):
                    label = class_names[cls_id]
                
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # DEBUG IN TERMINAL: Arata ca suntem vii
        frame_count += 1
        if frame_count % 30 == 0: # Printeaza o data la 30 de cadre
            status = "DETECTAT" if detected_something else "Nimic"
            print(f"> Running... Frame {frame_count} [{status}]", flush=True)

        # 4. AFISARE
        # Aici e testul critic pentru fereastra
        cv2.imshow("Smart Bin Pi5 - TEST", frame)

        # Asteapta 1ms pentru tasta si pentru a procesa fereastra GUI
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(">>> [INFO] Utilizatorul a apasat Q. Iesire.", flush=True)
            break

except Exception as e:
    print(f"\n!!! [CRASH] Eroare in bucla while: {e}", flush=True)

finally:
    print(">>> [5/6] Oprire camera...", flush=True)
    picam2.stop()
    cv2.destroyAllWindows()
    print(">>> [6/6] Program terminat.", flush=True)