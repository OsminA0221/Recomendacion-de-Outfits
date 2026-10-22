import cv2
import numpy as np
from dataclasses import dataclass
import os
from typing import Optional
from ultralytics import YOLO

# Modelo de puntos del cuerpo. "yolov8n" es el más rápido pero impreciso con poca
# luz o ropa holgada; "yolov8s" es bastante más preciso y algo más lento.
# Si va muy lento en tu PC, vuelve a "yolov8n-pose.pt"; si aún falla, "yolov8m-pose.pt".
POSE_MODEL_NAME = "yolov8s-pose.pt"

# Confianza mínima (0 a 1) para aceptar cada punto.
#  - Si sale mucho "cuerpo incompleto" aunque te veas bien: baja HIP_MIN_CONF.
#  - Si las caderas se dibujan en lugares absurdos: súbela.
SHOULDER_MIN_CONF = 0.50
HIP_MIN_CONF = 0.35

# Plausibilidad: largo del torso (hombros->caderas) dividido entre el ancho
# de hombros. En personas reales ronda 1.0-1.7; fuera de este rango los
# puntos de cadera casi seguro están mal ubicados.
TORSO_TO_SHOULDER_MIN = 0.8
TORSO_TO_SHOULDER_MAX = 2.4
# Hombros o caderas no pueden estar mas desnivelados que esta fraccion del torso.
MAX_TILT_FRACTION = 0.35

# Las mediciones se hacen en cortes HORIZONTALES de la silueta, así que se
# usa la altura media de cada par de puntos (hombros y caderas quedan a nivel).
# Evita que el ruido del modelo haga ver un hombro más alto que el otro.
LEVEL_SHOULDERS_AND_HIPS = True

# Suavizado entre frames de los puntos (1.0 = sin suavizado, 0.5 = medio).
KEYPOINT_SMOOTHING = 0.5

# Mejora de poca luz: si la imagen es oscura se aplica CLAHE + gamma ANTES de
# pasarla a los modelos (no modifica el video que ves). Subir el umbral = se
# activa más a menudo (0-255, brillo medio).
ENHANCE_LOW_LIGHT = True
LOW_LIGHT_MEAN = 100

# Manos en la cintura / codos abiertos ensanchan la silueta y falsean la cintura.
# Se rechaza el cuadro y se pide bajar los brazos. 0.6 = codo separado del
# hombro más de 60% del ancho de hombros. Pon False para desactivar.
REJECT_FLARED_ARMS = True
ARM_FLARE_MAX = 0.6

# Índices de puntos del modelo YOLO-Pose (formato COCO de 17 puntos).
KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_ANKLE, KP_R_ANKLE = 15, 16


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


def _iou(a, b) -> float:
    """Intersección sobre unión de dos cajas (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _enhance_low_light(frame_bgr: np.ndarray) -> np.ndarray:
    """Aclara imágenes oscuras para que YOLO vea mejor. Mismo tamaño que la original."""
    if not ENHANCE_LOW_LIGHT:
        return frame_bgr
    if float(np.mean(cv2.mean(frame_bgr)[:3])) >= LOW_LIGHT_MEAN:
        return frame_bgr

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if float(np.mean(cv2.mean(out)[:3])) < LOW_LIGHT_MEAN:
        lut = (((np.arange(256) / 255.0) ** 0.6) * 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
    return out


class PoseDetector:
    def __init__(self):
        # Dos modelos YOLO que se complementan:
        #   - yolov8n-seg:  silueta (máscara) de la persona -> mide los anchos.
        #   - yolov8n-pose: puntos reales del cuerpo (hombros, caderas...)
        #                   -> dice DÓNDE medir.
        # Se descargan automáticamente la primera vez (~6 MB cada uno).
        print("Cargando modelos Ultralytics YOLOv8-Seg + YOLOv8-Pose...")
        _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(_THIS_DIR, "..", "models")
        os.makedirs(MODEL_DIR, exist_ok=True)

        self.seg_model = YOLO(os.path.join(MODEL_DIR, "yolov8n-seg.pt"))
        self.pose_model = YOLO(os.path.join(MODEL_DIR, POSE_MODEL_NAME))
        self.model = self.seg_model   # alias por compatibilidad

        self._last_segmentation_mask = None
        self._last_landmarks_px = None
        self.last_issue = ""            # motivo legible del último rechazo
        self._prev_kxy = None           # para suavizar puntos entre frames
        self._prev_conf = None

    def _invisible(self, reason: str = "no se ven bien hombros y caderas") -> BodyLandmarks:
        self._last_landmarks_px = []   # para que no se dibujen puntos
        self.last_issue = reason
        self._prev_kxy = None
        self._prev_conf = None
        z = (0, 0)
        return BodyLandmarks(z, z, z, z, z, z, z, visibility_ok=False)

    def process(self, frame_bgr: np.ndarray) -> Optional[BodyLandmarks]:
        h_orig, w_orig = frame_bgr.shape[:2]
        self.last_issue = ""
        frame_in = _enhance_low_light(frame_bgr)   # solo para los modelos

        # 1) SEGMENTACIÓN: máscara de la persona principal
        seg = self.seg_model(frame_in, classes=[0], verbose=False)[0]

        if (seg.boxes is None or len(seg.boxes) == 0
                or seg.masks is None or len(seg.masks) == 0):
            self._last_segmentation_mask = None
            self._last_landmarks_px = None
            self._prev_kxy = None
            self._prev_conf = None
            self.last_issue = "no se detecta una persona"
            return None

        seg_boxes = seg.boxes.xyxy.cpu().numpy()
        areas = (seg_boxes[:, 2] - seg_boxes[:, 0]) * (seg_boxes[:, 3] - seg_boxes[:, 1])
        main_idx = int(np.argmax(areas))
        main_box = seg_boxes[main_idx]

        mask_2d = np.zeros((h_orig, w_orig), dtype=np.uint8)
        contour = np.array(seg.masks.xy[main_idx], dtype=np.int32)
        if len(contour) > 0:
            cv2.fillPoly(mask_2d, [contour], 1)
        self._last_segmentation_mask = mask_2d.astype(np.float32)

        # 2) POSE: puntos reales del cuerpo de ESA misma persona
        pose = self.pose_model(frame_in, verbose=False)[0]

        if (pose.keypoints is None or pose.boxes is None
                or len(pose.boxes) == 0):
            return self._invisible()

        pose_boxes = pose.boxes.xyxy.cpu().numpy()
        # Entre las personas de pose, tomamos la que mejor coincide con
        # la máscara elegida (evita mezclar a dos personas distintas).
        best = max(range(len(pose_boxes)), key=lambda i: _iou(pose_boxes[i], main_box))
        if _iou(pose_boxes[best], main_box) < 0.3:
            return self._invisible("no coinciden la silueta y los puntos del cuerpo")

        kxy = pose.keypoints.xy.cpu().numpy()[best]              # (17, 2) en px
        kconf_t = pose.keypoints.conf
        kconf = (kconf_t.cpu().numpy()[best] if kconf_t is not None
                 else np.ones(len(kxy)))

        raw_kxy = kxy.copy()   # sin suavizar, para validar

        # Suavizado temporal de los puntos (solo los que tienen confianza).
        if (KEYPOINT_SMOOTHING < 1.0 and self._prev_kxy is not None
                and self._prev_conf is not None):
            ok_pts = (kconf >= 0.3) & (self._prev_conf >= 0.3)
            kxy = np.where(
                ok_pts[:, None],
                KEYPOINT_SMOOTHING * kxy + (1.0 - KEYPOINT_SMOOTHING) * self._prev_kxy,
                kxy,
            )
        self._prev_kxy = kxy.copy()
        self._prev_conf = np.array(kconf, dtype=float)

        # 3) VALIDACIÓN: ¿se ven bien hombros y caderas?
        if any(kconf[i] < SHOULDER_MIN_CONF for i in (KP_L_SHOULDER, KP_R_SHOULDER)):
            return self._invisible("no se ven bien los hombros (mas luz, de frente)")
        if any(kconf[i] < HIP_MIN_CONF for i in (KP_L_HIP, KP_R_HIP)):
            return self._invisible("no se ven bien las caderas (mas luz, muestra hasta medio muslo)")

        shoulder_y = (kxy[KP_L_SHOULDER][1] + kxy[KP_R_SHOULDER][1]) / 2
        hip_y = (kxy[KP_L_HIP][1] + kxy[KP_R_HIP][1]) / 2
        torso_px = hip_y - shoulder_y

        # Las caderas deben estar debajo de los hombros y el torso debe
        # tener un tamaño mínimo (si no, la persona está demasiado lejos).
        if torso_px < h_orig * 0.10:
            return self._invisible("acercate: el torso se ve muy pequeno")

        # PLAUSIBILIDAD: descarta puntos de cadera mal ubicados (muy arriba,
        # sobre las rodillas, desnivelados...) en lugar de medir con ellos.
        shoulder_w_px = abs(kxy[KP_R_SHOULDER][0] - kxy[KP_L_SHOULDER][0])
        if shoulder_w_px < 1:
            return self._invisible("no se ven bien los hombros (mas luz, de frente)")
        torso_ratio = torso_px / shoulder_w_px
        if not (TORSO_TO_SHOULDER_MIN <= torso_ratio <= TORSO_TO_SHOULDER_MAX):
            return self._invisible("puntos del cuerpo inconsistentes (mejora la luz)")
        tilt_limit = MAX_TILT_FRACTION * torso_px
        if (abs(kxy[KP_L_HIP][1] - kxy[KP_R_HIP][1]) > tilt_limit
                or abs(kxy[KP_L_SHOULDER][1] - kxy[KP_R_SHOULDER][1]) > tilt_limit):
            return self._invisible("hombros o caderas muy desnivelados (ponte derecho)")

        # Manos en la cintura / codos abiertos: falsean el ancho de la cintura.
        if REJECT_FLARED_ARMS:
            for sh_i, other_i, elbow_i in (
                (KP_L_SHOULDER, KP_R_SHOULDER, KP_L_ELBOW),
                (KP_R_SHOULDER, KP_L_SHOULDER, KP_R_ELBOW),
            ):
                if kconf[elbow_i] < 0.4:
                    continue
                outward = 1.0 if raw_kxy[sh_i][0] > raw_kxy[other_i][0] else -1.0
                flare = (raw_kxy[elbow_i][0] - raw_kxy[sh_i][0]) * outward
                elbow_y = raw_kxy[elbow_i][1]
                if (flare > ARM_FLARE_MAX * shoulder_w_px
                        and shoulder_y <= elbow_y <= hip_y + 0.1 * torso_px):
                    return self._invisible("baja los brazos (no los pongas en la cintura)")

        # La medición de cadera escanea la región anatómica pélvica (~12% del torso
        # por debajo de los puntos de cadera) sin invadir los muslos.
        if hip_y + 0.12 * torso_px > h_orig * 0.99:
            return self._invisible("alejate un poco: debe verse la cadera completa")

        # 4) LANDMARKS NORMALIZADOS
        def norm(x, y):
            return (float(np.clip(x / w_orig, 0.0, 1.0)),
                    float(np.clip(y / h_orig, 0.0, 1.0)))

        l_sh, r_sh = kxy[KP_L_SHOULDER].copy(), kxy[KP_R_SHOULDER].copy()
        l_hp, r_hp = kxy[KP_L_HIP].copy(), kxy[KP_R_HIP].copy()

        if LEVEL_SHOULDERS_AND_HIPS:
            l_sh[1] = r_sh[1] = shoulder_y
            l_hp[1] = r_hp[1] = hip_y

        # La cintura no es un punto del modelo: se ubica a mitad de camino
        # entre hombro y cadera de cada lado (igual que hace la medición).
        l_waist = (l_sh + l_hp) / 2
        r_waist = (r_sh + r_hp) / 2

        if kconf[KP_NOSE] >= 0.3:
            nose_xy = kxy[KP_NOSE]
        else:
            nose_xy = np.array([(main_box[0] + main_box[2]) / 2, main_box[1]])

        def ankle(i):
            if kconf[i] >= 0.3:
                return kxy[i]
            return np.array([(main_box[0] + main_box[2]) / 2, main_box[3]])

        l_ank, r_ank = ankle(KP_L_ANKLE), ankle(KP_R_ANKLE)

        # Puntos en píxeles para dibujar (mismo orden que draw_landmarks)
        self._last_landmarks_px = [
            (int(p[0]), int(p[1])) for p in
            [l_sh, r_sh, l_waist, r_waist, l_hp, r_hp, nose_xy]
        ]

        return BodyLandmarks(
            left_shoulder=norm(*l_sh),
            right_shoulder=norm(*r_sh),
            left_hip=norm(*l_hp),
            right_hip=norm(*r_hp),
            left_ankle=norm(*l_ank),
            right_ankle=norm(*r_ank),
            nose=norm(*nose_xy),
            visibility_ok=True,
        )

    def _main_segment(self, y_normalized: float, threshold: float = 0.5):
        """Columnas (izquierda, derecha) del bloque central más grueso de la
        silueta en esa fila. Descarta partes separadas (brazos separados)."""
        mask = self._last_segmentation_mask
        if mask is None:
            return None

        h, w = mask.shape[:2]
        row_idx = int(np.clip(y_normalized, 0.0, 1.0) * (h - 1))
        row = mask[row_idx]

        person_cols = np.where(row >= threshold)[0]
        if person_cols.size < 2:
            return None

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

        return int(main_segment.min()), int(main_segment.max())

    def measure_silhouette_width(
        self,
        y_normalized: float,
        threshold: float = 0.5
    ) -> Optional[float]:
        """Mide el ancho usando los bordes de la máscara YOLO, ignorando los brazos."""
        seg = self._main_segment(y_normalized, threshold)
        if seg is None:
            return None
        w = self._last_segmentation_mask.shape[1]
        return float(seg[1] - seg[0]) / w

    def silhouette_edges(
        self,
        y_normalized: float,
        threshold: float = 0.5
    ):
        """Extremos izquierdo y derecho de la silueta en una fila.

        Devuelve ((x_izq, y), (x_der, y)) normalizados, o None.
        Son los puntos EXACTOS entre los que se mide el ancho.
        """
        seg = self._main_segment(y_normalized, threshold)
        if seg is None:
            return None
        w = self._last_segmentation_mask.shape[1]
        y = float(np.clip(y_normalized, 0.0, 1.0))
        return ((seg[0] / w, y), (seg[1] / w, y))

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
        """Dibuja las articulaciones del modelo como puntos pequeños y grises.

        El esqueleto "de verdad" (extremos de hombros, cintura y cadera sobre
        la silueta) lo dibuja overlay.draw_results.
        """
        pts = self._last_landmarks_px
        if not pts:
            return frame_bgr

        for pt in pts:
            cv2.circle(frame_bgr, pt, 3, (170, 170, 170), -1)

        return frame_bgr

    def get_segmentation_mask(self) -> Optional[np.ndarray]:
        return self._last_segmentation_mask

    def close(self) -> None:
        pass