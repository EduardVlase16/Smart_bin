import time
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# --- CONFIGURARE UTILIZATOR ---
CONFIDENCE_THRESHOLD = 0.5  # Ignoră ce e sub 50% sigur
CONSECUTIVE_FRAMES_TRIGGER = 5 # Câte cadre la rând trebuie să vadă același obiect ca să sorteze
COOLDOWN_SECONDS = 3.0 # Pauză după o sortare (să aibă timp motorul să revină)

# Mapează numele claselor din YOLO la ID-ul coșului tău
# (Verifică numele exact din modelul tău best.pt!)
CATEGORY_MAP = {
    "plastic": 1,
    "metal": 2,
    "cardboard": 3,
    "waste": 0
}
DEFAULT_BIN = 0 # Unde merge gunoiul nerecunoscut

# Lista claselor pentru afișare (doar vizual)
DISPLAY_CLASSES = ["Cardboard", "Metal", "Plastic", "Waste"]

# --- INITIALIZARE HARDWARE ---
print(">>> [INIT] Pornire Camera (BGR Format)...", flush=True)
try:
    picam2 = Picamera2()
    # TRUCUL PENTRU CULORI: Cerem direct BGR888
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()
except Exception as e:
    print(f"!!! Eroare Camera: {e}")
    sys.exit()

print(">>> [INIT] Incarcare Model...", flush=True)
try:
    model = YOLO("../weights/best.pt") # Asigură-te de cale!
    # model = YOLO("yolov8n.pt") # Fallback pt teste
except Exception as e:
    print(f"!!! Eroare Model: {e}")
    sys.exit()

# --- ZONA DE CONTROL SERVO ---
def trigger_servo_action(category_name):
    """
    AICI LIPESTI CODUL TAU PENTRU MOTOARE.
    Aceasta functie este apelata o singura data cand s-a luat decizia.
    """
    bin_id = CATEGORY_MAP.get(category_name.lower(), DEFAULT_BIN)
    
    print(f"\n✅ ACTIUNE DECLANSATA: Sortare [{category_name}] -> BIN {bin_id}")
    
    # --- EXEMPLU LOGICA SERVO (Pseudocod) ---
    # move_carousel(bin_id)
    # open_flap()
    # time.sleep(1)
    # close_flap()
    # return_home()
    # ----------------------------------------
    
    return True # Returneaza True cand a terminat miscarea

# --- LOOP PRINCIPAL ---
print(">>> [READY] Sistem pregatit. Astept gunoi...", flush=True)

consecutive_count = 0
last_seen_class = None
is_sorting = False
last_sort_time = 0

try:
    while True:
        # 0. Verificam cooldown (daca tocmai a sortat, stam putin)
        if time.time() - last_sort_time < COOLDOWN_SECONDS:
            # Continuam sa citim camera ca sa nu se blocheze bufferul, dar nu procesam AI
            picam2.capture_array() 
            continue

        # 1. Captura (Direct BGR acum!)
        frame = picam2.capture_array()

        # 2. Inferenta AI
        results = model(frame, stream=True, verbose=False)

        # 3. Gasirea celui mai bun obiect din cadru
        best_class = None
        max_conf = 0.0
        current_box = None # Pentru desenare

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > CONFIDENCE_THRESHOLD and conf > max_conf:
                    max_conf = conf
                    cls_id = int(box.cls[0])
                    # Extragem numele clasei din dictionarul modelului sau lista noastra
                    if hasattr(model, 'names'):
                        best_class = model.names[cls_id]
                    elif cls_id < len(DISPLAY_CLASSES):
                        best_class = DISPLAY_CLASSES[cls_id]
                    current_box = box

        # 4. Logica de Stabilizare (Debounce)
        if best_class:
            # Desenam obiectul detectat
            x1, y1, x2, y2 = map(int, current_box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{best_class} {max_conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Verificam stabilitatea
            if best_class == last_seen_class:
                consecutive_count += 1
            else:
                consecutive_count = 1 # Reset daca s-a schimbat obiectul
                last_seen_class = best_class

            # 5. DECLANSARE ACTIUNE
            if consecutive_count >= CONSECUTIVE_FRAMES_TRIGGER:
                print(f"> Confirmare stabila: {best_class}")
                
                # Apelam functia de servo
                trigger_servo_action(best_class)
                
                # Resetam totul si intram in cooldown
                consecutive_count = 0
                last_seen_class = None
                last_sort_time = time.time()
                print(">>> Pauza racire (Cooldown)...")

        else:
            # Nu se vede nimic relevant
            consecutive_count = 0
            last_seen_class = None

        # 6. Afisare
        cv2.imshow("Smart Bin Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Oprire manuala.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Sistem oprit.")