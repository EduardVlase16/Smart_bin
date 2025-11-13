Ghid de Configurare: SmartBin AI Detector (Partea de Training și Detecție)Acest ghid te va ajuta să configurezi mediul de lucru pe PC-ul tău pentru a antrena modelul YOLOv8 și a rula detectorul de obiecte, urmând pașii din proiectul Instructables.Pasul 1: Structura ProiectuluiÎți recomand să creezi un folder principal pentru proiect (de exemplu, SmartBin-Detector) și să-l deschizi în VSCode. În interior, vei avea următoarea structură:SmartBin-Detector/
├── .venv/               # Mediul tău virtual Python (îl vom crea)
├── data/                # Aici vor fi descărcate datele de la Roboflow
├── training/            # Scripturile pentru antrenare
│   ├── train.py           # (Opțional) Pentru antrenare locală
│   └── training_colab.py  # (Recomandat) Cod pentru Google Colab
├── detector/            # Scriptul pentru detecție live
│   └── detect.py
├── weights/             # Aici vei salva modelul antrenat (ex. best.pt)
└── README.md            # Acest fișier

# Pasul 2: Configurare Mediu Virtual (Virtual Environment)

Este esențial să folosești un mediu virtual pentru a nu strica alte proiecte Python.
Deschide folderul SmartBin-Detector în VSCode.Deschide un terminal nou în VSCode (Terminal > New Terminal).
Rulează comanda pentru a crea mediul virtual:python -m venv .venv
## Activează mediul virtual:Pe Windows (PowerShell/CMD):
.\.venv\Scripts\activate
## Pe macOS/Linux (bash):source .venv/bin/activate
Vei vedea (.venv) apărând la începutul liniei de terminal.

# Pasul 3: Instalarea DependențelorCu mediul activat, instalează librăriile necesare. 

Proiectul folosește YOLOv8 (care vine prin pachetul ultralytics), PyTorch (dependință pentru YOLO) și OpenCV (pentru a folosi camera web).Rulează în terminal:pip install ultralytics opencv-python torch torchvision

# Pasul 4: Colectarea și Adnotarea Datelor (Pașii 3-4 din Instructables)

Acest pas este manual și se face în afara VSCode.Colectează imagini: Fă poze la diverse tipuri de gunoi (plastic, hârtie, metal, etc.), exact ca în ghid.Adnotare cu Roboflow:Creează un cont gratuit pe Roboflow.Creează un proiect nou (alege "Object Detection").Încarcă imaginile tale.Treci prin fiecare imagine și desenează casete (bounding boxes) în jurul obiectelor, alocând eticheta corectă (ex: "plastic", "paper").Când ai terminat, apasă "Generate" pentru a crea o versiune a setului tău de date.La pasul de "Export", alege formatul "YOLOv8". Roboflow îți va oferi un link sau un fragment de cod pentru a descărca datele.Pasul 5: Antrenarea Modelului (Pasul 5 din Instructables)Ai două opțiuni. Îți recomand cu tărie opțiunea 1 (Google Colab), deoarece antrenarea AI necesită un GPU puternic, iar Colab îți oferă unul gratuit.Opțiunea 1: Antrenare pe Google Colab (Recomandat)Deschide Google Colab și creează un "New Notebook".Asigură-te că folosești un GPU: Runtime > Change runtime type > Hardware accelerator > T4 GPU.Copiază celulele din fișierul training/training_colab.py în notebook-ul tău Colab și rulează-le pe rând.La finalul antrenării, vei avea un fișier numit best.pt în folderul runs/detect/train/weights/.Descarcă acest fișier best.pt de pe Colab și salvează-l pe PC-ul tău în folderul weights/.Opțiunea 2: Antrenare Locală (Opțional - Doar dacă ai un GPU NVIDIA)Dacă ai un PC cu un GPU NVIDIA puternic și ai instalat CUDA, poți încerca să antrenezi local.Descarcă datele de pe Roboflow (format YOLOv8) și dezarhivează-le în folderul data/. Vei obține un fișier data.yaml.Editează fișierul training/train.py pentru a pune calea corectă către fișierul tău data.yaml.Rulează scriptul din terminal:python training/train.py
La final, modelul best.pt se va afla în runs/detect/train/weights/. Copiază-l în folderul weights/.Pasul 6: Rularea Detectorului (Pasul 7 din Instructables)Acum că ai modelul antrenat (best.pt) în folderul weights/, poți rula detectorul pe camera ta web.Editează fișierul detector/detect.py.IMPORTANT: Trebuie să actualizezi lista class_names pentru a corespunde exact cu etichetele pe care le-ai creat în Roboflow (ex: ['plastic', 'paper', 'metal']).Rulează scriptul din terminal (asigură-te că mediul .venv e activat):python detector/detect.py
Apasă tasta 'q' pentru a opri camera.


# Cod pentru a descarca datasetul folosit la antrenare

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="0IvyJ3dhBaYzm81uf99D")
project = rf.workspace("datasetforpython").project("dataset-labeling-nmkkk")
version = project.version(1)
dataset = version.download("yolov8")
                