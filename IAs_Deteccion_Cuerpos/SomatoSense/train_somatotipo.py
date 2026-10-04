import random
import numpy as np
import torch
from ultralytics import YOLO

# -----------------------------------------------------------
# 1. REPRODUCIBILIDAD COMPLETA
# -----------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -----------------------------------------------------------
# 2. CARGAR MODELO BASE
# -----------------------------------------------------------
model = YOLO("yolov8n.pt")   # Puedes cambiar a yolov8s.pt si tu GPU lo permite

# -----------------------------------------------------------
# 3. ENTRENAMIENTO
# -----------------------------------------------------------
model.train(
    #data=r"C:\Users\Gabriel\Desktop\proyecto IA\Proyecto Somatotipos.v2-80-20-dataset-yolov8\data.yaml",
    data=r"C:\Users\Gabriel\Desktop\proyecto IA\Proyecto Somatotipos 2.v1i\data.yaml",
    
    epochs=100,                 # recomendado
    patience=30,                # early stopping recomendado
    imgsz=640,
    batch=8,
    device="cpu",                   # 0=GPU, "cpu" si no tienes
    workers=2,
    seed=42,
    optimizer="Adam",           # mejor convergencia para datasets pequeños
    cos_lr=True,                # scheduler suave y eficiente
    name="somatotipo_final_v3",
    project=r"C:\Users\Gabriel\Desktop\proyecto IA\runs"
)
