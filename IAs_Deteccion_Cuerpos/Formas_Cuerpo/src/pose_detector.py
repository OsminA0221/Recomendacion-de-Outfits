import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

#Este es el modelo de pose landmarker lite de MediaPipe, que es mas rapido y liviano que el modelo completo. Se descarga automaticamente la primera vez que se ejecuta el programa.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_THIS_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")

# Umbral de confianza de la mascara de segmentacion para considerar que
# un pixel pertenece a la persona (0.0 a 1.0).
SEGMENTATION_THRESHOLD = 0.5


def _ensure_model_downloaded() -> str:
    """Descarga el modelo de pose la primera vez que se necesita."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo de deteccion de pose (una sola vez, ~5-10 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Modelo guardado en: {MODEL_PATH}")
    return MODEL_PATH


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


# Conexiones simplificadas para dibujar un esqueleto basico sobre el
# video. Los indices son los mismos 33 landmarks estandar de MediaPipe
# Pose (0=nariz, 11/12=hombros, 23/24=cadera, etc.).
_SKELETON_CONNECTIONS = [
    (11, 12),          # linea de hombros
    (11, 23), (12, 24),  # torso
    (23, 24),          # linea de cadera
    (11, 13), (13, 15),  # brazo izquierdo
    (12, 14), (14, 16),  # brazo derecho
    (23, 25), (25, 27),  # pierna izquierda
    (24, 26), (26, 28),  # pierna derecha
    (0, 11), (0, 12),   # cabeza a hombros
]


class PoseDetector:
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    # Umbral minimo de "visibility" que reporta MediaPipe para considerar
    # un punto confiable (0.0 a 1.0).
    MIN_VISIBILITY = 0.5

    def __init__(self, min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.6) -> None:
        model_path = _ensure_model_downloaded()

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=True,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0
        self._last_landmarks_px = None      # cache para dibujar el esqueleto
        self._last_segmentation_mask = None  # cache para medir la silueta

    def process(self, frame_bgr: np.ndarray) -> Optional[BodyLandmarks]:
        """Procesa un frame y devuelve los landmarks relevantes.

        Devuelve None si no se detectó una persona en el frame.
        """
        rgb_frame = np.ascontiguousarray(frame_bgr[:, :, ::-1])  # BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # La API de video requiere timestamps crecientes; usamos un
        # contador propio en vez de tiempo real para evitar problemas si
        # el procesamiento es mas lento que la camara.
        self._frame_timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        if result.segmentation_masks:
            self._last_segmentation_mask = result.segmentation_masks[0].numpy_view()
        else:
            self._last_segmentation_mask = None

        if not result.pose_landmarks:
            self._last_landmarks_px = None
            return None

        lm = result.pose_landmarks[0]  # primera persona detectada
        h, w = frame_bgr.shape[:2]
        self._last_landmarks_px = [(int(p.x * w), int(p.y * h)) for p in lm]

        def point(idx: int) -> tuple:
            p = lm[idx]
            return (p.x, p.y)

        def visible(idx: int) -> bool:
            return lm[idx].visibility >= self.MIN_VISIBILITY

        required_points = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_HIP, self.RIGHT_HIP,
            self.LEFT_ANKLE, self.RIGHT_ANKLE,
        ]
        visibility_ok = all(visible(idx) for idx in required_points)

        return BodyLandmarks(
            left_shoulder=point(self.LEFT_SHOULDER),
            right_shoulder=point(self.RIGHT_SHOULDER),
            left_hip=point(self.LEFT_HIP),
            right_hip=point(self.RIGHT_HIP),
            left_ankle=point(self.LEFT_ANKLE),
            right_ankle=point(self.RIGHT_ANKLE),
            nose=point(self.NOSE),
            visibility_ok=visibility_ok,
        )
    def measure_silhouette_width(
        self,
        y_normalized: float,
        threshold: float = SEGMENTATION_THRESHOLD
        ) -> Optional[float]:
        """
        Mide el ancho de la silueta en una línea horizontal.

        Se utiliza la máscara de segmentación de MediaPipe.
        Se intenta conservar el segmento principal de la persona
        para reducir errores producidos por píxeles aislados.
        """

        mask = self._last_segmentation_mask

        if mask is None:
            return None

        h, w = mask.shape[:2]

        row_idx = int(
            np.clip(
                y_normalized,
                0.0,
                1.0
            ) * (h - 1)
        )

        row = mask[row_idx]

        person_cols = np.where(
            row >= threshold
        )[0]

        if person_cols.size < 2:
            return None

        # ----------------------------------------------------------
        # Buscar el segmento continuo más grande.
        # ----------------------------------------------------------

        gaps = np.diff(person_cols)

        split_positions = np.where(
            gaps > 3
        )[0]

        segments = []

        start = 0

        for split in split_positions:

            segment = person_cols[
                start:split + 1
            ]

            if segment.size > 0:
                segments.append(segment)

            start = split + 1

        # Último segmento.

        segment = person_cols[start:]

        if segment.size > 0:
            segments.append(segment)

        if not segments:
            return None

        # Elegir el segmento más grande.
        main_segment = max(
            segments,
            key=len
        )

        if main_segment.size < 2:
            return None

        left = main_segment.min()
        right = main_segment.max()

        return float(
            right - left
        ) / w
    
    def measure_silhouette_width_range(self, y_start: float, y_end: float,
                                        mode: str = "max",
                                        threshold: float = SEGMENTATION_THRESHOLD,
                                        samples: int = 15) -> Optional[float]:
        """Escanea varias alturas entre y_start y y_end y devuelve el ancho
        MAXIMO o MINIMO de silueta encontrado en ese rango."""
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
        """Dibuja un esqueleto simplificado sobre el frame."""
        pts = self._last_landmarks_px
        if not pts:
            return frame_bgr

        for a, b in _SKELETON_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 0), 2)

        key_points = (self.NOSE, self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
                      self.LEFT_HIP, self.RIGHT_HIP,
                      self.LEFT_ANKLE, self.RIGHT_ANKLE)
        for idx in key_points:
            if idx < len(pts):
                cv2.circle(frame_bgr, pts[idx], 5, (0, 0, 255), -1)

        return frame_bgr

    def close(self) -> None:
        self._landmarker.close()