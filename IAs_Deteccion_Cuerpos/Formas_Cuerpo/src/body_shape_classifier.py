from dataclasses import dataclass

from src.body_measurements import BodyProportions

# Umbrales calibrados para silueta frontal 2D con ropa ligera/casual:
# Nota geométrica clave: en vista frontal 2D se cumple WHR = SHR * WSR.
# Cuando los hombros son más anchos que la cadera (SHR > 1.08) o cuando una
# prenda cae recta en un cuerpo Rectángulo/Trapecio, WHR suele rondar 0.90-0.98
# SIN que exista abdomen prominente. Un cuerpo Óvalo/Manzana real requiere que
# la cintura/abdomen iguale o supere TANTO a la cadera como a los hombros o que
# sobresalga convexamente respecto a la línea hombro-cadera (torso_convexity).

# Mujer
MUJER_SHOULDER_HIP_MARGIN = 0.08
MUJER_WAIST_DEFINED_WHR = 0.83      # en ancho 2D con ropa equivale a ~0.75 de perímetro 3D
MUJER_WAIST_DEFINED_WSR = 0.84
MUJER_OVAL_WHR_MIN = 0.99           # cintura prácticamente igual o más ancha que cadera...
MUJER_OVAL_WSR_MIN = 0.95           # ...y también casi igual o más ancha que hombros
MUJER_OVAL_CONVEXITY = 1.045        # o curvatura abdominal convexa clara hacia afuera

# Hombre
HOMBRE_TRAPEZOID_SHR_MIN = 1.06
HOMBRE_VSHAPE_SHR_MIN = 1.20
HOMBRE_RECTANGLE_MARGIN = 0.06
HOMBRE_OVAL_WHR_MIN = 1.02          # cintura más ancha que cadera...
HOMBRE_OVAL_WSR_MIN = 0.96          # ...y prácticamente tan ancha o más que los hombros
HOMBRE_OVAL_CONVEXITY = 1.055       # o curvatura abdominal convexa clara hacia afuera


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
    """Devuelve una forma corporal estimada a partir de las proporciones 2D y el género."""

    shr = proportions.shoulder_to_hip_ratio
    whr = proportions.waist_to_hip_ratio

    wsr = getattr(proportions, "waist_to_shoulder_ratio", 0.0)
    if wsr <= 0.0 and shr > 1e-6:
        wsr = whr / shr

    convexity = getattr(proportions, "torso_convexity", 0.0)
    if convexity <= 0.0 and (shr + 1.0) > 1e-6:
        convexity = whr / (0.5 * shr + 0.5)

    is_male = str(gender).strip().lower() in ("hombre", "masculino", "man", "male", "h")

    if is_male:
        normalized_gender = "Hombre"
        es_ovalo = (
            (whr >= HOMBRE_OVAL_WHR_MIN and wsr >= HOMBRE_OVAL_WSR_MIN)
            or (convexity >= HOMBRE_OVAL_CONVEXITY and whr >= 0.93 and wsr >= 0.89)
        )

        # 1. Óvalo: prominencia real y convexa en la zona media respecto a hombros y cadera
        if es_ovalo:
            label = "Ovalo"
        # 2. Triángulo: cadera notoriamente más ancha que hombros (sin prominencia convexa central)
        elif shr < 1.0 - HOMBRE_RECTANGLE_MARGIN:
            label = "Triangulo"
        # 3. Triángulo invertido: espalda en V marcada y cintura estrecha respecto a hombros
        elif shr >= HOMBRE_VSHAPE_SHR_MIN and wsr <= 0.89:
            label = "Triangulo invertido"
        # 4. Trapecio: hombros más anchos que cadera con caída atlética hacia cintura
        elif shr >= HOMBRE_TRAPEZOID_SHR_MIN and wsr <= 0.93:
            label = "Trapecio"
        # 5. Rectángulo: hombros, cintura y cadera alineados en caída recta
        else:
            label = "Rectangulo"

        note = (
            f"Clasificación masculina basada en hombro/cadera ({shr:.2f}), "
            f"cintura/cadera ({whr:.2f}) y cintura/hombro ({wsr:.2f})."
        )
    else:
        normalized_gender = "Mujer"
        balanced = abs(shr - 1.0) <= MUJER_SHOULDER_HIP_MARGIN
        waist_defined = (whr <= MUJER_WAIST_DEFINED_WHR) and (wsr <= MUJER_WAIST_DEFINED_WSR)
        es_manzana = (
            (whr >= MUJER_OVAL_WHR_MIN and wsr >= MUJER_OVAL_WSR_MIN)
            or (convexity >= MUJER_OVAL_CONVEXITY and whr >= 0.91 and wsr >= 0.88)
        )

        # 1. Ovalado / Manzana: zona media genuinamente más ancha o convexa que hombros y cadera
        if es_manzana:
            label = "Ovalado / Manzana"
        # 2. Reloj de arena: hombros y cadera equilibrados con cintura claramente marcada
        elif balanced and waist_defined:
            label = "Reloj de arena"
        # 3. Triángulo invertido: hombros visiblemente más anchos que la cadera
        elif shr > 1.0 + MUJER_SHOULDER_HIP_MARGIN:
            label = "Triangulo invertido"
        # 4. Triángulo / Pera: caderas visiblemente más anchas que los hombros
        elif shr < 1.0 - MUJER_SHOULDER_HIP_MARGIN:
            label = "Triangulo / Pera"
        # 5. Rectángulo: proporciones equilibradas y caída de torso recta
        else:
            label = "Rectangulo"

        note = (
            f"Clasificación femenina basada en hombro/cadera ({shr:.2f}), "
            f"cintura/cadera ({whr:.2f}) y cintura/hombro ({wsr:.2f})."
        )

    return BodyShapeResult(
        label=label,
        shoulder_to_hip_ratio=round(shr, 3),
        waist_to_hip_ratio=round(whr, 3),
        gender=normalized_gender,
        confidence_note=note,
    )
