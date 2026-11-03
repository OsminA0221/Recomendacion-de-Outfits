from dataclasses import dataclass, field
import numpy as np
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

    # Extremos izquierdo/derecho de la silueta en cada fila medida:
    # ((x_izq, y), (x_der, y)) normalizados, o None si no se midió.
    shoulder_edges: Optional[tuple] = None
    waist_edges: Optional[tuple] = None
    hip_edges: Optional[tuple] = None

    # Métricas geométricas adicionales para distinguir torso recto/atlético de óvalo real:
    # - waist_to_shoulder_ratio: cintura / hombros (clave cuando hombros > cadera).
    # - torso_convexity: relación máxima entre el ancho medio/abdominal y la recta
    #   hombro-cadera (>1.05 indica prominencia convexa real; <=1.01 indica caída recta o cóncava).
    waist_to_shoulder_ratio: float = 0.0
    torso_convexity: float = 1.0


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
    Mantiene estabilidad temporal de las medidas principales y la convexidad del torso.
    """

    shoulder: TemporalSmoother
    waist: TemporalSmoother
    hip: TemporalSmoother
    convexity: TemporalSmoother = field(default_factory=lambda: TemporalSmoother(0.20))

    @classmethod
    def create(cls, alpha: float = 0.20):
        return cls(
            shoulder=TemporalSmoother(alpha),
            waist=TemporalSmoother(alpha),
            hip=TemporalSmoother(alpha),
            convexity=TemporalSmoother(alpha),
        )

    def reset(self):
        self.shoulder.reset()
        self.waist.reset()
        self.hip.reset()
        self.convexity.reset()


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
    de la máscara sea interpretada como hombro, cintura o cadera.
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


def _measure_shoulder_width_2d(
    detector: PoseDetector,
    shoulder_y: float,
    torso_height: float
) -> tuple[Optional[float], Optional[float]]:
    """
    Mide el ancho bi-deltoideo justo en la corona del hombro (shoulder_y a +5% del torso)
    usando la mediana local para no inflar hombros si los brazos se abren más abajo.
    """
    start_y = shoulder_y - (torso_height * 0.01)
    end_y = shoulder_y + (torso_height * 0.05)
    target_y = shoulder_y + (torso_height * 0.02)

    measurements = _sample_widths(
        detector,
        start_y,
        end_y,
        samples_count=9
    )

    if len(measurements) < 3:
        w = detector.measure_silhouette_width(shoulder_y)
        return (w, shoulder_y) if w is not None else (None, None)

    smoothed = _smooth_width_curve(measurements, window=3)
    widths_sorted = sorted(w for _, w in smoothed)
    median_width = widths_sorted[len(widths_sorted) // 2]

    best_y, _ = min(smoothed, key=lambda item: (abs(item[1] - median_width), abs(item[0] - target_y)))
    return median_width, best_y


def _measure_waist_width_2d(
    detector: PoseDetector,
    shoulder_y: float,
    hip_y: float,
    torso_height: float
) -> tuple[Optional[float], Optional[float]]:
    """
    Busca la cintura natural dentro de la región anatómica entre costillas bajas
    y cresta ilíaca superior (36% a 78% del torso), evitando bajar hasta la cadera
    en cuerpos en V donde el mínimo global caería sobre la propia pelvis.
    """

    start_y = shoulder_y + torso_height * 0.43
    end_y = shoulder_y + torso_height * 0.78
    if end_y >= hip_y - torso_height * 0.12:
        end_y = hip_y - torso_height * 0.12

    measurements = _sample_widths(
        detector,
        start_y,
        end_y,
        samples_count=19
    )

    if len(measurements) < 5:
        return None, None

    measurements = _smooth_width_curve(
        measurements,
        window=3
    )

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
        return None, None

    avg_width = sum(
        width for _, width in neighborhood
    ) / len(neighborhood)

    return avg_width, measurements[min_index][0]


def _compute_torso_convexity(
    detector: PoseDetector,
    shoulder_y: float,
    torso_height: float,
    shoulder_width: float,
    hip_width: float,
) -> float:
    """
    Mide si la zona media/abdominal sobresale de forma CONVEXA respecto a la
    línea recta que une el ancho de hombros con el ancho de cadera.

    - <= 1.00 : silueta cóncava o recta (reloj de arena, trapecio, rectángulo con camiseta).
    - >= 1.05 : prominencia convexa real en abdomen/zona media (óvalo / manzana).
    """
    if shoulder_width <= 0 or hip_width <= 0 or torso_height <= 0:
        return 1.0

    start_y = shoulder_y + torso_height * 0.44
    end_y = shoulder_y + torso_height * 0.84
    raw = _sample_widths(detector, start_y, end_y, samples_count=15)
    if len(raw) < 5:
        return 1.0

    smoothed = _smooth_width_curve(raw, window=5)
    ratios = []
    for y, w in smoothed:
        t = float(np.clip((y - shoulder_y) / torso_height, 0.0, 1.0))
        expected_linear = (1.0 - t) * shoulder_width + t * hip_width
        if expected_linear > 1e-6:
            ratios.append(w / expected_linear)

    if not ratios:
        return 1.0

    # Percentil alto suavizado para evitar picos aislados de un solo renglón
    ratios.sort()
    top_slice = ratios[max(0, len(ratios) - 3):]
    return float(sum(top_slice) / len(top_slice))


def _measure_true_hip_2d(
    detector: PoseDetector,
    pelvis_y: float,
    torso_height: float
) -> tuple[Optional[float], Optional[float]]:
    """
    Escanea la región anatómica de la cadera (cresta ilíaca a trocánter mayor)
    para encontrar el ancho máximo representativo de la cadera/pelvis,
    evitando descender hacia los muslos o la entrepierna.
    """

    # Escaneo centrado en la pelvis/hueso de la cadera:
    # -8% por encima de los puntos de cadera (cresta ilíaca)
    # +12% por debajo (trocánteres mayores / curva superior glútea)
    start_y = pelvis_y - (torso_height * 0.08)
    end_y = pelvis_y + (torso_height * 0.12)

    measurements = _sample_widths(
        detector,
        start_y,
        end_y,
        samples_count=21
    )

    if len(measurements) < 3:
        w = detector.measure_silhouette_width(pelvis_y)
        return (w, pelvis_y) if w is not None else (None, None)

    smoothed = _smooth_width_curve(
        measurements,
        window=3
    )

    max_width = max(w for _, w in smoothed)

    # Para evitar que baje hacia los muslos si la silueta es recta o la ropa es holgada:
    # consideramos los puntos dentro del 97% del máximo y elegimos el más cercano anatómicamente a pelvis_y.
    candidates = [
        (y, w) for y, w in smoothed if w >= max_width * 0.97
    ]

    best_y, _ = min(
        candidates,
        key=lambda item: abs(item[0] - pelvis_y)
    )

    best_idx = min(
        range(len(smoothed)),
        key=lambda i: abs(smoothed[i][0] - best_y)
    )

    neighborhood_radius = 2
    start_idx = max(0, best_idx - neighborhood_radius)
    end_idx = min(len(smoothed), best_idx + neighborhood_radius + 1)
    neighborhood = smoothed[start_idx:end_idx]

    avg_width = sum(w for _, w in neighborhood) / len(neighborhood)

    return avg_width, best_y


def compute_proportions(
    landmarks: BodyLandmarks,
    detector: Optional[PoseDetector] = None,
    smoother: Optional[BodyMeasurementSmoother] = None
) -> BodyProportions:
    """
    Calcula las proporciones corporales utilizando un modelo 2D.

    El sistema utiliza:

    - landmarks para localizar regiones anatómicas y aislar los brazos;
    - segmentación para obtener los anchos reales del torso/pelvis;
    - varias secciones horizontales con suavizado espacial y temporal;
    - convexidad de la zona media respecto a la línea hombro-cadera.

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

    raw_shoulder_y = shoulder_mid[1]
    raw_hip_y = hip_mid[1]

    # La cintura no tiene landmark propio.
    waist_y = (
        raw_shoulder_y + raw_hip_y
    ) / 2

    torso_height = _euclidean(
        shoulder_mid,
        hip_mid
    )

    estimated_total_height = _euclidean(
        landmarks.nose,
        ankle_mid
    )

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
    shoulder_y = raw_shoulder_y
    hip_y = raw_hip_y
    torso_convexity = 0.92

    measured_from_silhouette = False
    shoulder_edges = waist_edges = hip_edges = None

    if detector is not None:

        silhouette_shoulder, new_shoulder_y = _measure_shoulder_width_2d(
            detector,
            raw_shoulder_y,
            torso_height
        )

        silhouette_waist, waist_row_y = _measure_waist_width_2d(
            detector,
            raw_shoulder_y,
            raw_hip_y,
            torso_height
        )

        silhouette_hip, new_hip_y = _measure_true_hip_2d(
            detector,
            raw_hip_y,
            torso_height
        )

        if None not in (
            silhouette_shoulder,
            silhouette_hip,
            silhouette_waist
        ):
            # El borde externo deltoideo nunca puede ser menor que la distancia entre centros articulares
            shoulder_width = max(silhouette_shoulder, fallback_shoulder_width * 1.04)
            hip_width = silhouette_hip
            waist_width = silhouette_waist

            shoulder_y = new_shoulder_y
            hip_y = new_hip_y
            waist_y = waist_row_y

            torso_convexity = _compute_torso_convexity(
                detector,
                raw_shoulder_y,
                torso_height,
                shoulder_width,
                hip_width,
            )

            shoulder_edges = detector.silhouette_edges(shoulder_y)
            waist_edges = detector.silhouette_edges(waist_row_y)
            hip_edges = detector.silhouette_edges(new_hip_y)

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

        torso_convexity = (
            smoother.convexity.update(
                torso_convexity
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

    waist_to_shoulder_ratio = _safe_ratio(
        waist_width,
        shoulder_width
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

        shoulder_edges=shoulder_edges,
        waist_edges=waist_edges,
        hip_edges=hip_edges,

        waist_to_shoulder_ratio=waist_to_shoulder_ratio,
        torso_convexity=torso_convexity,
    )


def _safe_ratio(
    numerator: float,
    denominator: float
) -> float:

    if denominator == 0:
        return 0.0

    return numerator / denominator
