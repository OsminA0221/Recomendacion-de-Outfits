import random
import numpy as np
import torch
from ultralytics import YOLO

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

model = YOLO("yolov8n.pt")   # Puedes cambiar a yolov8s.pt si tu GPU lo permite

model.train(
    data=r"C:\Users\Gabriel\Desktop\proyecto IA\Proyecto Somatotipos 2.v1i\data.yaml",
    
    epochs=100,
    patience=30,
    imgsz=640,
    batch=8,
    device="cpu",                   # 0=GPU, "cpu" si no tienes
    workers=2,
    seed=42,
    optimizer="Adam",
    cos_lr=True,
    name="somatotipo_final_v3",
    project=r"C:\Users\Gabriel\Desktop\proyecto IA\runs"
)
