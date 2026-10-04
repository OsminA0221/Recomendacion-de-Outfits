from dataclasses import dataclass
from typing import Optional

from src.pose_detector import BodyLandmarks, PoseDetector


def _euclidean(p1: tuple, p2: tuple) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _midpoint(p1: tuple, p2: tuple) -> tuple:
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


@dataclass
class BodyProportions:
    shoulder_width: float
    hip_width: float
    torso_height: float
    estimated_total_height: float

    shoulder_to_hip_ratio: float
    torso_to_height_ratio: float

    estimated_waist_width: float
    waist_to_hip_ratio: float

    # Posiciones verticales normalizadas utilizadas para las mediciones.
    shoulder_y: float
    waist_y: float
    hip_y: float

    # Indica si las medidas proceden de la silueta.
    measured_from_silhouette: bool


class TemporalSmoother:
    """
    Suaviza una medición entre frames.

    alpha pequeño:
        más estable, pero responde más lento.

    alpha grande:
        responde más rápido, pero puede tener más movimiento.
    """

    def __init__(self, alpha: float = 0.20):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, new_value: Optional[float]) -> Optional[float]:

        if new_value is None:
            return self.value

        if self.value is None:
            self.value = new_value
            return self.value

        self.value = (
            self.alpha * new_value
            + (1.0 - self.alpha) * self.value
        )

        return self.value

    def reset(self) -> None:
        self.value = None


@dataclass
class BodyMeasurementSmoother:
    """
    Mantiene estabilidad temporal de las tres medidas principales.
    """

    shoulder: TemporalSmoother
    waist: TemporalSmoother
    hip: TemporalSmoother

    @classmethod
    def create(cls, alpha: float = 0.20):
        return cls(
            shoulder=TemporalSmoother(alpha),
            waist=TemporalSmoother(alpha),
            hip=TemporalSmoother(alpha),
        )

    def reset(self):
        self.shoulder.reset()
        self.waist.reset()
        self.hip.reset()


def _sample_widths(
    detector: PoseDetector,
    start_y: float,
    end_y: float,
    samples_count: int = 21
) -> list[tuple[float, float]]:
    """
    Obtiene múltiples mediciones de ancho de silueta.

    Devuelve:
        [(y, width), ...]
    """

    measurements = []

    if samples_count < 2:
        samples_count = 2

    for i in range(samples_count):

        t = i / (samples_count - 1)

        y = start_y + (end_y - start_y) * t

        width = detector.measure_silhouette_width(y)

        if width is not None and width > 0:
            measurements.append((y, width))

    return measurements


def _smooth_width_curve(
    measurements: list[tuple[float, float]],
    window: int = 3
) -> list[tuple[float, float]]:
    """
    Suaviza espacialmente la curva ancho-Y.

    Esto evita que un solo píxel o una pequeña irregularidad
    de la máscara sea interpretada como la cadera.
    """

    if len(measurements) < 3:
        return measurements

    values = [width for _, width in measurements]

    smoothed = []

    half = window // 2

    for i in range(len(values)):

        start = max(0, i - half)
        end = min(len(values), i + half + 1)

        average = sum(values[start:end]) / (end - start)

        smoothed.append(
            (measurements[i][0], average)
        )

    return smoothed


def _measure_hip_width_2d(
    detector: PoseDetector,
    hip_y: float,
    torso_height: float
) -> Optional[float]:
    """
    Detecta el ancho de cadera utilizando la silueta 2D.

    No utiliza simplemente max(widths).

    En su lugar:

    1. Define una región alrededor de la cadera.
    2. Obtiene múltiples cortes horizontales.
    3. Suaviza la curva.
    4. Encuentra el máximo de la curva suavizada.
    5. Promedia varios puntos alrededor de ese máximo.

    Esto reduce considerablemente los saltos causados por
    ruido de segmentación.
    """

    # Región anatómica de búsqueda.

    start_y = hip_y - torso_height * 0.08
    end_y = hip_y + torso_height * 0.22

    measurements = _sample_widths(
        detector,
        start_y,
        end_y,
        samples_count=25
    )

    if len(measurements) < 7:
        return None

    measurements = _smooth_width_curve(
        measurements,
        window=3
    )

    if not measurements:
        return None

    # Encontrar el punto más ancho de la curva suavizada.

    peak_index = max(
        range(len(measurements)),
        key=lambda i: measurements[i][1]
    )

    # En vez de utilizar únicamente el máximo,
    # tomamos varios puntos alrededor del máximo.

    neighborhood_radius = 2

    start_index = max(
        0,
        peak_index - neighborhood_radius
    )

    end_index = min(
        len(measurements),
        peak_index + neighborhood_radius + 1
    )

    neighborhood = measurements[
        start_index:end_index
    ]

    if len(neighborhood) < 3:
        return None

    # Promedio de la zona más ancha.

    return sum(
        width for _, width in neighborhood
    ) / len(neighborhood)


def _measure_waist_width_2d(
    detector: PoseDetector,
    shoulder_y: float,
    hip_y: float,
    torso_height: float
) -> Optional[float]:
    """
    Busca la cintura como una zona estrecha dentro de la región
    entre hombros y cadera.

    Se evita utilizar un único mínimo porque un error aislado
    de segmentación podría generar una cintura artificialmente
    pequeña.
    """

    start_y = shoulder_y + torso_height * 0.35
    end_y = hip_y - torso_height * 0.05

    measurements = _sample_widths(
        detector,
        start_y,
        end_y,
        samples_count=21
    )

    if len(measurements) < 7:
        return None

    measurements = _smooth_width_curve(
        measurements,
        window=3
    )

    # Buscar el punto más estrecho de la curva suavizada.

    min_index = min(
        range(len(measurements)),
        key=lambda i: measurements[i][1]
    )

    neighborhood_radius = 2

    start_index = max(
        0,
        min_index - neighborhood_radius
    )

    end_index = min(
        len(measurements),
        min_index + neighborhood_radius + 1
    )

    neighborhood = measurements[
        start_index:end_index
    ]

    if len(neighborhood) < 3:
        return None

    return sum(
        width for _, width in neighborhood
    ) / len(neighborhood)


def compute_proportions(
    landmarks: BodyLandmarks,
    detector: Optional[PoseDetector] = None,
    smoother: Optional[BodyMeasurementSmoother] = None
) -> BodyProportions:
    """
    Calcula las proporciones corporales utilizando un modelo 2D.

    El sistema utiliza:

    - landmarks para localizar regiones anatómicas;
    - segmentación para obtener los anchos reales de la silueta;
    - varias secciones horizontales;
    - suavizado espacial;
    - suavizado temporal.

    Si la silueta no está disponible, utiliza landmarks como respaldo.
    """

    shoulder_mid = _midpoint(
        landmarks.left_shoulder,
        landmarks.right_shoulder
    )

    hip_mid = _midpoint(
        landmarks.left_hip,
        landmarks.right_hip
    )

    ankle_mid = _midpoint(
        landmarks.left_ankle,
        landmarks.right_ankle
    )

    shoulder_y = shoulder_mid[1]
    hip_y = hip_mid[1]

    # La cintura no tiene landmark propio.
    waist_y = (
        shoulder_y + hip_y
    ) / 2

    torso_height = _euclidean(
        shoulder_mid,
        hip_mid
    )

    estimated_total_height = _euclidean(
        landmarks.nose,
        ankle_mid
    )

    # ==========================================================
    # RESPALDO MEDIANTE LANDMARKS
    # ==========================================================

    fallback_shoulder_width = _euclidean(
        landmarks.left_shoulder,
        landmarks.right_shoulder
    )

    fallback_hip_width = _euclidean(
        landmarks.left_hip,
        landmarks.right_hip
    )

    fallback_waist_width = (
        (fallback_shoulder_width + fallback_hip_width) / 2
    ) * 0.85

    shoulder_width = fallback_shoulder_width
    hip_width = fallback_hip_width
    waist_width = fallback_waist_width

    measured_from_silhouette = False

    # ==========================================================
    # MEDICIONES MEDIANTE SILUETA
    # ==========================================================

    if detector is not None:

        silhouette_shoulder = (
            detector.measure_silhouette_width(
                shoulder_y
            )
        )

        silhouette_hip = _measure_hip_width_2d(
            detector,
            hip_y,
            torso_height
        )

        silhouette_waist = _measure_waist_width_2d(
            detector,
            shoulder_y,
            hip_y,
            torso_height
        )

        if None not in (
            silhouette_shoulder,
            silhouette_hip,
            silhouette_waist
        ):

            shoulder_width = silhouette_shoulder
            hip_width = silhouette_hip
            waist_width = silhouette_waist

            measured_from_silhouette = True

    if smoother is not None:

        shoulder_width = (
            smoother.shoulder.update(
                shoulder_width
            )
        )

        waist_width = (
            smoother.waist.update(
                waist_width
            )
        )

        hip_width = (
            smoother.hip.update(
                hip_width
            )
        )

    shoulder_to_hip_ratio = _safe_ratio(
        shoulder_width,
        hip_width
    )

    waist_to_hip_ratio = _safe_ratio(
        waist_width,
        hip_width
    )

    torso_to_height_ratio = _safe_ratio(
        torso_height,
        estimated_total_height
    )

    return BodyProportions(
        shoulder_width=shoulder_width,
        hip_width=hip_width,
        torso_height=torso_height,
        estimated_total_height=estimated_total_height,

        shoulder_to_hip_ratio=shoulder_to_hip_ratio,
        torso_to_height_ratio=torso_to_height_ratio,

        estimated_waist_width=waist_width,
        waist_to_hip_ratio=waist_to_hip_ratio,

        shoulder_y=shoulder_y,
        waist_y=waist_y,
        hip_y=hip_y,

        measured_from_silhouette=measured_from_silhouette,
    )


def _safe_ratio(
    numerator: float,
    denominator: float
) -> float:

    if denominator == 0:
        return 0.0

    return numerator / denominator