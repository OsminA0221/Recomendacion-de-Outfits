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
KP_L_WRIST, KP_R_WRIST = 9, 10
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
        #   - yolov8s-pose: puntos reales del cuerpo (hombros, codos, muñecas, caderas...)
        #                   -> dice DÓNDE medir y permite separar los brazos del torso.
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
        self._spine_pts = None          # ((sh_mid_x, sh_y), (hp_mid_x, hp_y)) en px
        self.last_issue = ""            # motivo legible del último rechazo
        self._prev_kxy = None           # para suavizar puntos entre frames
        self._prev_conf = None

    def _invisible(self, reason: str = "no se ven bien hombros y caderas") -> BodyLandmarks:
        self._last_landmarks_px = []   # para que no se dibujen puntos
        self._spine_pts = None
        self.last_issue = reason
        self._prev_kxy = None
        self._prev_conf = None
        z = (0, 0)
        return BodyLandmarks(z, z, z, z, z, z, z, visibility_ok=False)

    @staticmethod
    def _carve_arms_from_torso(
        mask_2d: np.ndarray,
        kxy: np.ndarray,
        kconf: np.ndarray,
        shoulder_y: float,
        hip_y: float,
        torso_px: float,
        shoulder_w_px: float,
    ) -> None:
        """Separa los brazos colgantes de la máscara del torso por debajo de la axila.

        yolov8n-seg genera máscaras de baja resolución que suelen pegar los brazos
        y mangas a la cintura/cadera, haciendo que la cintura mida de codo a codo
        y clasificando erróneamente casi todo cuerpo como 'Óvalo'.
        """
        h, w = mask_2d.shape[:2]
        y_cut = int(np.clip(shoulder_y + 0.24 * torso_px, 0, h - 1))
        if y_cut >= h - 1:
            return

        sh_mid_x = 0.5 * (kxy[KP_L_SHOULDER][0] + kxy[KP_R_SHOULDER][0])
        hp_mid_x = 0.5 * (kxy[KP_L_HIP][0] + kxy[KP_R_HIP][0])
        torso_mid_x = 0.5 * (sh_mid_x + hp_mid_x)

        # Radio anatómico estimado del brazo respecto al ancho bi-articular de hombros.
        # En vez de borrar un tubo grueso (que podría morder cintura o cadera), trazamos
        # una hendidura fina de separación (4 px) justo en el pliegue interno del brazo
        # (arm_center - 0.85 * r_arm); luego _main_segment toma únicamente el bloque
        # central del torso e ignora por completo el brazo separado.
        r_arm = max(5.0, 0.125 * shoulder_w_px)
        slit_offset = 0.82 * r_arm

        # Bordes externos reales a nivel del hombro para anclar el eje superior del brazo
        sh_row_idx = int(np.clip(shoulder_y, 0, h - 1))
        sh_cols = np.where(mask_2d[sh_row_idx] > 0)[0]
        sh_left_edge = float(sh_cols[0]) if sh_cols.size >= 2 else float(min(kxy[KP_L_SHOULDER][0], kxy[KP_R_SHOULDER][0]))
        sh_right_edge = float(sh_cols[-1]) if sh_cols.size >= 2 else float(max(kxy[KP_L_SHOULDER][0], kxy[KP_R_SHOULDER][0]))

        arm_chains = (
            (KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_L_ELBOW, KP_L_WRIST),
            (KP_R_SHOULDER, KP_L_SHOULDER, KP_R_HIP, KP_R_ELBOW, KP_R_WRIST),
        )

        for sh_i, other_sh_i, hp_i, el_i, wr_i in arm_chains:
            if kconf[el_i] < 0.30:
                continue
            outward = 1.0 if kxy[sh_i][0] > kxy[other_sh_i][0] else -1.0
            el_pt = kxy[el_i]

            # Solo separar el brazo si cuelga hacia afuera del eje central
            if (el_pt[0] - torso_mid_x) * outward < 0.35 * shoulder_w_px:
                continue

            side_mask = np.zeros_like(mask_2d)

            # Centro del brazo superior a nivel del deltoides (no la articulación clavicular interna)
            sh_edge = sh_right_edge if outward > 0 else sh_left_edge
            sh_arm_cx = sh_edge - outward * (0.72 * r_arm)
            sh_crease = np.array([sh_arm_cx - outward * slit_offset, shoulder_y])
            el_crease = np.array([el_pt[0] - outward * slit_offset, el_pt[1]])

            cv2.line(
                side_mask,
                (int(round(sh_crease[0])), int(round(sh_crease[1]))),
                (int(round(el_crease[0])), int(round(el_crease[1]))),
                1,
                thickness=4,
            )

            if kconf[wr_i] >= 0.30:
                wr_pt = kxy[wr_i]
                if (wr_pt[0] - torso_mid_x) * outward >= 0.30 * shoulder_w_px:
                    wr_crease = np.array([wr_pt[0] - outward * (0.78 * r_arm), wr_pt[1]])
                    hand_crease = wr_crease + 0.45 * (wr_crease - el_crease)
                    cv2.line(
                        side_mask,
                        (int(round(el_crease[0])), int(round(el_crease[1]))),
                        (int(round(hand_crease[0])), int(round(hand_crease[1]))),
                        1,
                        thickness=4,
                    )
            else:
                down_crease = np.array([el_crease[0], min(h - 1, hip_y + 0.20 * torso_px)])
                cv2.line(
                    side_mask,
                    (int(round(el_crease[0])), int(round(el_crease[1]))),
                    (int(round(down_crease[0])), int(round(down_crease[1]))),
                    1,
                    thickness=4,
                )

            # Proteger el núcleo del torso por regiones anatómicas:
            # - En cintura (arriba de la cresta ilíaca): resguardo estrecho para permitir
            #   separar brazos pegados incluso en siluetas Pera / Reloj de arena.
            # - En cadera/pelvis: resguardo más amplio anclado a la articulación de cadera
            #   para que muñecas/manos nunca recorten el ancho real de cadera.
            hip_zone_y = int(np.clip(hip_y - 0.12 * torso_px, y_cut, h - 1))
            waist_guard_x = int(round(0.32 * shoulder_w_px))
            hip_guard_x = int(round(
                max(
                    (kxy[hp_i][0] - torso_mid_x) * outward + 0.14 * shoulder_w_px,
                    0.36 * shoulder_w_px,
                )
            ))
            if outward > 0:
                lim_waist = min(w, int(round(torso_mid_x)) + waist_guard_x)
                lim_hip = min(w, int(round(torso_mid_x)) + hip_guard_x)
                side_mask[y_cut:hip_zone_y, :lim_waist] = 0
                side_mask[hip_zone_y:, :lim_hip] = 0
            else:
                lim_waist = max(0, int(round(torso_mid_x)) - waist_guard_x)
                lim_hip = max(0, int(round(torso_mid_x)) - hip_guard_x)
                side_mask[y_cut:hip_zone_y, lim_waist:] = 0
                side_mask[hip_zone_y:, lim_hip:] = 0

            mask_2d[y_cut:][side_mask[y_cut:] > 0] = 0

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
            self._spine_pts = None
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

        # 2) POSE: puntos reales del cuerpo de ESA misma persona
        pose = self.pose_model(frame_in, verbose=False)[0]

        if (pose.keypoints is None or pose.boxes is None
                or len(pose.boxes) == 0):
            self._last_segmentation_mask = mask_2d.astype(np.float32)
            return self._invisible()

        pose_boxes = pose.boxes.xyxy.cpu().numpy()
        # Entre las personas de pose, tomamos la que mejor coincide con
        # la máscara elegida (evita mezclar a dos personas distintas).
        best = max(range(len(pose_boxes)), key=lambda i: _iou(pose_boxes[i], main_box))
        if _iou(pose_boxes[best], main_box) < 0.3:
            self._last_segmentation_mask = mask_2d.astype(np.float32)
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
            self._last_segmentation_mask = mask_2d.astype(np.float32)
            return self._invisible("no se ven bien los hombros (mas luz, de frente)")
        if any(kconf[i] < HIP_MIN_CONF for i in (KP_L_HIP, KP_R_HIP)):
            self._last_segmentation_mask = mask_2d.astype(np.float32)
            return self._invisible("no se ven bien las caderas (mas luz, muestra hasta medio muslo)")

        shoulder_y = (kxy[KP_L_SHOULDER][1] + kxy[KP_R_SHOULDER][1]) / 2
        hip_y = (kxy[KP_L_HIP][1] + kxy[KP_R_HIP][1]) / 2
        torso_px = hip_y - shoulder_y

        # Las caderas deben estar debajo de los hombros y el torso debe
        # tener un tamaño mínimo (si no, la persona está demasiado lejos).
        if torso_px < h_orig * 0.10:
            self._last_segmentation_mask = mask_2d.astype(np.float32)
            return self._invisible("acercate: el torso se ve muy pequeno")

        # PLAUSIBILIDAD: descarta puntos de cadera mal ubicados (muy arriba,
        # sobre las rodillas, desnivelados...) en lugar de medir con ellos.
        shoulder_w_px = abs(kxy[KP_R_SHOULDER][0] - kxy[KP_L_SHOULDER][0])
        if shoulder_w_px < 1:
            self._last_segmentation_mask = mask_2d.astype(np.float32)
            return self._invisible("no se ven bien los hombros (mas luz, de frente)")
        torso_ratio = torso_px / shoulder_w_px
        if not (TORSO_TO_SHOULDER_MIN <= torso_ratio <= TORSO_TO_SHOULDER_MAX):
            self._last_segmentation_mask = mask_2d.astype(np.float32)
            return self._invisible("puntos del cuerpo inconsistentes (mejora la luz)")
        tilt_limit = MAX_TILT_FRACTION * torso_px
        if (abs(kxy[KP_L_HIP][1] - kxy[KP_R_HIP][1]) > tilt_limit
                or abs(kxy[KP_L_SHOULDER][1] - kxy[KP_R_SHOULDER][1]) > tilt_limit):
            self._last_segmentation_mask = mask_2d.astype(np.float32)
            return self._invisible("hombros o caderas muy desnivelados (ponte derecho)")

        torso_mid_x = 0.25 * (
            kxy[KP_L_SHOULDER][0] + kxy[KP_R_SHOULDER][0]
            + kxy[KP_L_HIP][0] + kxy[KP_R_HIP][0]
        )

        # Manos en la cintura / codos abiertos / brazos cruzados frente al abdomen:
        if REJECT_FLARED_ARMS:
            for sh_i, other_i, elbow_i, wrist_i in (
                (KP_L_SHOULDER, KP_R_SHOULDER, KP_L_ELBOW, KP_L_WRIST),
                (KP_R_SHOULDER, KP_L_SHOULDER, KP_R_ELBOW, KP_R_WRIST),
            ):
                if kconf[elbow_i] >= 0.4:
                    outward = 1.0 if raw_kxy[sh_i][0] > raw_kxy[other_i][0] else -1.0
                    flare = (raw_kxy[elbow_i][0] - raw_kxy[sh_i][0]) * outward
                    elbow_y = raw_kxy[elbow_i][1]
                    if (flare > ARM_FLARE_MAX * shoulder_w_px
                            and shoulder_y <= elbow_y <= hip_y + 0.1 * torso_px):
                        self._last_segmentation_mask = mask_2d.astype(np.float32)
                        return self._invisible("baja los brazos (no los pongas en la cintura)")

                if kconf[wrist_i] >= 0.45:
                    wx, wy = raw_kxy[wrist_i]
                    if (shoulder_y + 0.25 * torso_px <= wy <= hip_y + 0.05 * torso_px
                            and abs(wx - torso_mid_x) < 0.32 * shoulder_w_px):
                        self._last_segmentation_mask = mask_2d.astype(np.float32)
                        return self._invisible("deja los brazos sueltos a los lados")

        # La medición de cadera escanea la región anatómica pélvica (~12% del torso
        # por debajo de los puntos de cadera) sin invadir los muslos.
        if hip_y + 0.12 * torso_px > h_orig * 0.99:
            self._last_segmentation_mask = mask_2d.astype(np.float32)
            return self._invisible("alejate un poco: debe verse la cadera completa")

        # Separar brazos colgantes de la silueta del torso antes de medir cintura y cadera
        self._carve_arms_from_torso(
            mask_2d, kxy, kconf, shoulder_y, hip_y, torso_px, shoulder_w_px
        )
        self._last_segmentation_mask = mask_2d.astype(np.float32)

        sh_mid_x = 0.5 * (kxy[KP_L_SHOULDER][0] + kxy[KP_R_SHOULDER][0])
        hp_mid_x = 0.5 * (kxy[KP_L_HIP][0] + kxy[KP_R_HIP][0])
        self._spine_pts = ((sh_mid_x, shoulder_y), (hp_mid_x, hip_y))

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

    def _expected_center_col(self, row_idx: int, w: int) -> Optional[int]:
        """Columna X esperada del eje central del torso en la fila dada."""
        if self._spine_pts is None:
            return None
        (sx, sy), (hx, hy) = self._spine_pts
        if abs(hy - sy) < 1.0:
            return int(np.clip(0.5 * (sx + hx), 0, w - 1))
        t = float(np.clip((row_idx - sy) / (hy - sy), -0.2, 1.3))
        cx = sx + t * (hx - sx)
        return int(np.clip(round(cx), 0, w - 1))

    def _main_segment(self, y_normalized: float, threshold: float = 0.5):
        """Columnas (izquierda, derecha) del bloque del torso en esa fila.
        Separa cualquier hueco real (>2-3 px) entre brazo y cintura y prioriza
        el segmento que contiene el eje central del cuerpo."""
        mask = self._last_segmentation_mask
        if mask is None:
            return None

        h, w = mask.shape[:2]
        row_idx = int(np.clip(y_normalized, 0.0, 1.0) * (h - 1))
        row = mask[row_idx]

        person_cols = np.where(row >= threshold)[0]
        if person_cols.size < 2:
            return None

        # Antes estaba fijo en gaps > 20, lo que unía brazos separados hasta por 20 px
        # de aire con la cintura. Ahora cualquier separación real (>2-3 px) divide.
        max_gap = max(2, int(round(w * 0.003)))
        gaps = np.diff(person_cols)
        split_positions = np.where(gaps > max_gap)[0]

        segments = []
        start = 0
        for split in split_positions:
            segment = person_cols[start:split + 1]
            if segment.size >= 2:
                segments.append(segment)
            start = split + 1

        segment = person_cols[start:]
        if segment.size >= 2:
            segments.append(segment)

        if not segments:
            return None

        center_col = self._expected_center_col(row_idx, w)
        if center_col is not None:
            # Elegir el segmento que cubre el eje espinal (o el más cercano a él si es suficientemente ancho)
            containing = [s for s in segments if s[0] <= center_col <= s[-1]]
            if containing:
                main_segment = max(containing, key=len)
                return int(main_segment[0]), int(main_segment[-1])

            viable = [s for s in segments if len(s) >= max(6, int(w * 0.02))]
            if viable:
                main_segment = min(
                    viable,
                    key=lambda s: min(abs(int(s[0]) - center_col), abs(int(s[-1]) - center_col)),
                )
                return int(main_segment[0]), int(main_segment[-1])

        main_segment = max(segments, key=len)
        return int(main_segment[0]), int(main_segment[-1])

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
