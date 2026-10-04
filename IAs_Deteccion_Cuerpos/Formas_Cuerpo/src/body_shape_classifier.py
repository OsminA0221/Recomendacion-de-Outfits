from dataclasses import dataclass

from src.body_measurements import BodyProportions

# Umbrales de decisión. Son valores iniciales razonables para un MVP;
# se recomienda calibrarlos con datos reales (por ejemplo, un pequeño
# set de imágenes etiquetadas manualmente) antes de pasar a producción.
SHOULDER_HIP_BALANCED_MARGIN = 0.05
WAIST_DEFINED_THRESHOLD = 0.75


@dataclass
class BodyShapeResult:
    label: str
    shoulder_to_hip_ratio: float
    waist_to_hip_ratio: float
    confidence_note: str


def classify_body_shape(proportions: BodyProportions) -> BodyShapeResult:
    """Devuelve una forma corporal estimada a partir de las proporciones."""

    shr = proportions.shoulder_to_hip_ratio
    whr = proportions.waist_to_hip_ratio

    balanced = abs(shr - 1.0) <= SHOULDER_HIP_BALANCED_MARGIN
    waist_defined = whr <= WAIST_DEFINED_THRESHOLD

    if balanced and waist_defined:
        label = "Reloj de arena"
    elif shr > 1.0 + SHOULDER_HIP_BALANCED_MARGIN:
        label = "Triangulo invertido"
    elif shr < 1.0 - SHOULDER_HIP_BALANCED_MARGIN:
        label = "Triangulo"
    else:
        label = "Rectangulo"

    return BodyShapeResult(
        label=label,
        shoulder_to_hip_ratio=round(shr, 3),
        waist_to_hip_ratio=round(whr, 3),
        confidence_note=(
            "Estimación por reglas geométricas (Fase 1). "
            "La cintura es aproximada, no medida directamente."
        ),
    )
