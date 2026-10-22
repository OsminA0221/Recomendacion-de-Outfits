from dataclasses import dataclass

from src.body_measurements import BodyProportions

# Umbrales de decisión para Mujer
MUJER_SHOULDER_HIP_MARGIN = 0.06
MUJER_WAIST_DEFINED_THRESHOLD = 0.75
MUJER_OVAL_WAIST_THRESHOLD = 0.88

# Umbrales de decisión para Hombre
HOMBRE_TRAPEZOID_SHR_MIN = 1.05
HOMBRE_TRAPEZOID_SHR_MAX = 1.22
HOMBRE_VSHAPE_SHR_MIN = 1.22
HOMBRE_RECTANGLE_MARGIN = 0.06
HOMBRE_OVAL_WHR_THRESHOLD = 0.95


@dataclass
class BodyShapeResult:
    label: str
    shoulder_to_hip_ratio: float
    waist_to_hip_ratio: float
    gender: str = "Mujer"
    confidence_note: str = ""


def classify_body_shape(
    proportions: BodyProportions,
    gender: str = "Mujer"
) -> BodyShapeResult:
    """Devuelve una forma corporal estimada a partir de las proporciones y el género especificado."""

    shr = proportions.shoulder_to_hip_ratio
    whr = proportions.waist_to_hip_ratio

    is_male = str(gender).strip().lower() in ("hombre", "masculino", "man", "male", "h")

    if is_male:
        normalized_gender = "Hombre"
        # 1. Óvalo: zona media/abdomen prominente
        if whr >= HOMBRE_OVAL_WHR_THRESHOLD or (whr >= 0.90 and shr < 1.02):
            label = "Ovalo"
        # 2. Triángulo: cadera más ancha que hombros
        elif shr < 1.0 - HOMBRE_RECTANGLE_MARGIN:
            label = "Triangulo"
        # 3. Triángulo invertido: hombros notablemente más anchos que cadera (espalda en V marcada)
        elif shr >= HOMBRE_VSHAPE_SHR_MIN:
            label = "Triangulo invertido"
        # 4. Trapecio: hombros moderadamente más anchos que cadera, cintura proporcionada (atlético estándar)
        elif HOMBRE_TRAPEZOID_SHR_MIN <= shr < HOMBRE_VSHAPE_SHR_MIN and whr < 0.90:
            label = "Trapecio"
        # 5. Rectángulo: hombros, cintura y cadera de anchura similar
        else:
            label = "Rectangulo"

        note = (
            f"Clasificación masculina basada en hombro/cadera ({shr:.2f}) "
            f"y cintura/cadera ({whr:.2f})."
        )
    else:
        normalized_gender = "Mujer"
        balanced = abs(shr - 1.0) <= MUJER_SHOULDER_HIP_MARGIN
        waist_defined = whr <= MUJER_WAIST_DEFINED_THRESHOLD

        # 1. Ovalado / Manzana: cintura notablemente ancha respecto a cadera
        if whr >= MUJER_OVAL_WAIST_THRESHOLD:
            label = "Ovalado / Manzana"
        # 2. Reloj de arena: hombros y cadera equilibrados con cintura definida
        elif balanced and waist_defined:
            label = "Reloj de arena"
        # 3. Triángulo invertido: hombros más anchos que cadera
        elif shr > 1.0 + MUJER_SHOULDER_HIP_MARGIN:
            label = "Triangulo invertido"
        # 4. Triángulo / Pera: caderas más anchas que hombros
        elif shr < 1.0 - MUJER_SHOULDER_HIP_MARGIN:
            label = "Triangulo / Pera"
        # 5. Rectángulo: hombros y cadera equilibrados pero cintura poco definida
        else:
            label = "Rectangulo"

        note = (
            f"Clasificación femenina basada en hombro/cadera ({shr:.2f}) "
            f"y cintura/cadera ({whr:.2f})."
        )

    return BodyShapeResult(
        label=label,
        shoulder_to_hip_ratio=round(shr, 3),
        waist_to_hip_ratio=round(whr, 3),
        gender=normalized_gender,
        confidence_note=note,
    )
