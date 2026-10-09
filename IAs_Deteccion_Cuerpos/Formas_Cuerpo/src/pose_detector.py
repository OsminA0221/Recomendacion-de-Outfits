import cv2
import numpy as np
from dataclasses import dataclass
import os
from typing import Optional
from ultralytics import YOLO

@dataclass
class BodyLandmarks:
    """Puntos clave relevantes para el cálculo de forma corporal.
    Cada punto es una tupla (x, y) normalizada en [0, 1].
    """
    left_shoulder: tuple
    right_shoulder: tuple
    left_hip: tuple
    right_hip: tuple
    left_ankle: tuple
    right_ankle: tuple
    nose: tuple
    visibility_ok: bool


class PoseDetector:
    def __init__(self):
        # Carga el modelo YOLOv8 Nano con segmentación.
        # Se descargará automáticamente la primera vez (~6 MB).
        print("Cargando modelo Ultralytics YOLOv8-Seg...")
        # 1. Definimos la ruta de la carpeta (la carpeta "models" un nivel atrás de "src")
        _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(_THIS_DIR, "..", "models")
        
        # 2. Nos aseguramos de que la carpeta exista
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # 3. Armamos la ruta completa para el archivo del modelo
        model_path = os.path.join(MODEL_DIR, "yolov8n-seg.pt")
        
        # 4. Le pasamos la ruta específica a YOLO
        self.model = YOLO(model_path)
        
        self._last_segmentation_mask = None
        self._last_landmarks_px = None

    def process(self, frame_bgr: np.ndarray) -> Optional[BodyLandmarks]:
        # Ejecutamos YOLO pidiendo solo la clase 0 (personas)
        results = self.model(frame_bgr, classes=[0], verbose=False)

        if not results or not results[0].boxes or not results[0].masks:
            self._last_segmentation_mask = None
            self._last_landmarks_px = None
            return None

        result = results[0]
        
        # 1. ENCONTRAR A LA PERSONA PRINCIPAL
        # Calculamos el área de todas las cajas para quedarnos con la más grande
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_idx = np.argmax(areas)

        # 2. GENERAR MÁSCARA A ESCALA ORIGINAL
        h_orig, w_orig = frame_bgr.shape[:2]
        mask_2d = np.zeros((h_orig, w_orig), dtype=np.uint8)
        contour = np.array(result.masks.xy[largest_idx], dtype=np.int32)
        
        if len(contour) > 0:
            cv2.fillPoly(mask_2d, [contour], 1)
        
        self._last_segmentation_mask = mask_2d.astype(np.float32)

        # 3. GENERAR LANDMARKS VIRTUALES PARA COMPATIBILIDAD
        x1, y1, x2, y2 = boxes[largest_idx]
        w_box = x2 - x1
        h_box = y2 - y1

        # ==============================================================
        # FILTRO: ¿Es un cuerpo o solo una cara?
        # Un cuerpo visible (del torso hacia abajo) tiene un alto mayor que su ancho.
        aspect_ratio = h_box / w_box
        
        # Si la imagen es casi cuadrada (cara) o no ocupa al menos el 35% 
        # de la pantalla, abortamos y ocultamos los puntos.
        if aspect_ratio < 1.3 or h_box < (h_orig * 0.35):
            self._last_landmarks_px = [] # Vaciamos los puntos para que no se dibujen
            return BodyLandmarks(
                left_shoulder=(0,0), right_shoulder=(0,0),
                left_hip=(0,0), right_hip=(0,0),
                left_ankle=(0,0), right_ankle=(0,0),
                nose=(0,0),
                visibility_ok=False # Esto le dirá a main.py que muestre la advertencia
            )
        # ==============================================================

        def norm(x, y):
            nx = np.clip(x / w_orig, 0.0, 1.0)
            ny = np.clip(y / h_orig, 0.0, 1.0)
            return (nx, ny)

        # Ubicamos los puntos usando proporciones estándar
        l_shoulder = norm(x1 + w_box * 0.21, y1 + h_box * 0.20)
        r_shoulder = norm(x1 + w_box * 0.79, y1 + h_box * 0.20)

        # Agregamos los puntos de la cintura (aprox al 35% de la altura de la caja)
        # Los hacemos un poco más estrechos (0.35 y 0.65) para dar forma al torso
        l_waist = norm(x1 + w_box * 0.35, y1 + h_box * 0.35)
        r_waist = norm(x1 + w_box * 0.65, y1 + h_box * 0.35) 
        
        l_hip = norm(x1 + w_box * 0.3, y1 + h_box * 0.50) # Mitad exacta
        r_hip = norm(x1 + w_box * 0.7, y1 + h_box * 0.50)
        
        l_ankle = norm(x1 + w_box * 0.4, y1 + h_box * 0.90)
        r_ankle = norm(x1 + w_box * 0.6, y1 + h_box * 0.90)
        
        nose = norm((x1 + x2) / 2, y1 + h_box * 0.05)

        # Guardamos los puntos en píxeles para dibujarlos luego
        self._last_landmarks_px = [
            (int(p[0]*w_orig), int(p[1]*h_orig)) for p in 
            [l_shoulder, r_shoulder, l_waist, r_waist, l_hip, r_hip, nose]
        ]

        return BodyLandmarks(
            left_shoulder=l_shoulder,
            right_shoulder=r_shoulder,
            left_hip=l_hip,
            right_hip=r_hip,
            left_ankle=l_ankle,
            right_ankle=r_ankle,
            nose=nose,
            visibility_ok=True
        )

    def measure_silhouette_width(
        self, 
        y_normalized: float, 
        threshold: float = 0.5
    ) -> Optional[float]:
        """Mide el ancho usando los bordes de la máscara YOLO, ignorando los brazos."""
        mask = self._last_segmentation_mask
        if mask is None:
            return None

        h, w = mask.shape[:2]
        row_idx = int(np.clip(y_normalized, 0.0, 1.0) * (h - 1))
        row = mask[row_idx]

        person_cols = np.where(row >= threshold)[0]
        if person_cols.size < 2:
            return None

        # Descartamos partes separadas (como los brazos separados del cuerpo)
        # buscando el bloque central más grueso
        gaps = np.diff(person_cols)
        split_positions = np.where(gaps > 20)[0] 

        segments = []
        start = 0
        for split in split_positions:
            segment = person_cols[start:split + 1]
            if segment.size > 0:
                segments.append(segment)
            start = split + 1

        segment = person_cols[start:]
        if segment.size > 0:
            segments.append(segment)

        if not segments:
            return None

        main_segment = max(segments, key=len)
        
        if main_segment.size < 2:
            return None

        left = main_segment.min()
        right = main_segment.max()
        
        return float(right - left) / w

    def measure_silhouette_width_range(
        self, 
        y_start: float, 
        y_end: float,
        mode: str = "max",
        threshold: float = 0.5,
        samples: int = 15
    ) -> Optional[float]:
        
        if self._last_segmentation_mask is None:
            return None

        y0, y1 = sorted((y_start, y_end))
        widths = []
        
        for i in range(samples):
            y = y0 + (y1 - y0) * (i / max(samples - 1, 1))
            width = self.measure_silhouette_width(y, threshold=threshold)
            if width is not None:
                widths.append(width)

        if not widths:
            return None

        return max(widths) if mode == "max" else min(widths)

    def draw_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Dibuja un esqueleto simplificado conectando los puntos virtuales."""
        pts = self._last_landmarks_px
        if not pts:
            return frame_bgr

        # Índices de los puntos guardados en self._last_landmarks_px:
        # 0: l_shoulder, 1: r_shoulder
        # 2: l_hip,      3: r_hip
        # 4: l_ankle,    5: r_ankle
        # 6: nose

        # Definir qué puntos se conectan con líneas
        connections = [
            (0, 1),  # Línea horizontal de hombros
            (2, 3),  # Línea horizontal de cintura
            (4, 5),  # Línea horizontal de cadera
            (0, 2),  # Torso superior izquierdo (Hombro -> Cintura)
            (1, 3),  # Torso superior derecho (Hombro -> Cintura)
            (2, 4),  # Torso inferior izquierdo (Cintura -> Cadera)
            (3, 5),  # Torso inferior derecho (Cintura -> Cadera)
            (6, 0),  # Cuello izquierdo
            (6, 1)   # Cuello derecho
        ]

        # 1. Dibujar las líneas del torso
        for a, b in connections:
            if a < len(pts) and b < len(pts):
                cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 0), 2)

        # 2. Dibujar los puntos guía
        for pt in pts:
            cv2.circle(frame_bgr, pt, 5, (0, 0, 255), -1)

        return frame_bgr

    def get_segmentation_mask(self) -> Optional[np.ndarray]:
        return self._last_segmentation_mask

    def close(self) -> None:
        pass