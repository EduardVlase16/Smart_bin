# !pip install roboflow --> COMANDA TERMINAL PENTRU INSTALAREA DATASETULUI

from roboflow import Roboflow
rf = Roboflow(api_key="0IvyJ3dhBaYzm81uf99D")
project = rf.workspace("datasetforpython").project("dataset-labeling-nmkkk")
version = project.version(1)
dataset = version.download("yolov8")
    