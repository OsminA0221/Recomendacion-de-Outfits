"""Utilidades de colorimetria y armonia cromatica.

- COLOR_HEX: asigna un codigo HEX representativo a cada colour_group_name del dataset.
- Conversion HEX -> RGB -> CIELAB y HSL para medir luminosidad perceptual (L*),
  croma (C*), matiz real (HSL + Lab), temperatura y distancia perceptual (Delta E).
- PerfilPiel: clasifica profundidad (clara, media, oscura), subtono (calido, frio,
  neutro) y estacion cromatica aproximada a partir del HEX detectado en el rostro.
- Puntajes de compatibilidad piel-prenda (considerando subtono, profundidad,
  saturacion y efecto lavado/camuflaje cerca del rostro) y armonia de outfit completo.
"""
from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------
# 1) Nombre de color del dataset -> HEX  (colour_group_name, 42 valores)
#    "Other" y "Unknown" no tienen color real: quedan en None.
# --------------------------------------------------------------------------
COLOR_HEX: dict[str, Optional[str]] = {
    "Beige": "#D8C3A5", "Black": "#111111", "Blue": "#2E5DA8",
    "Dark Beige": "#A38A63", "Dark Blue": "#1F2A5A", "Dark Green": "#1E4D2B",
    "Dark Grey": "#4A4A4A", "Dark Orange": "#B5521B", "Dark Pink": "#B03A6A",
    "Dark Purple": "#4B2A6B", "Dark Red": "#7A1420", "Dark Turquoise": "#0F6B73",
    "Dark Yellow": "#B8940F", "Green": "#2E8B57", "Greenish Khaki": "#7A7A4A",
    "Grey": "#808080", "Greyish Beige": "#B5AA98", "Light Beige": "#EADFC8",
    "Light Blue": "#9EC5E8", "Light Green": "#A8D5A2", "Light Grey": "#C8C8C8",
    "Light Orange": "#F6B98A", "Light Pink": "#F4C2CF", "Light Purple": "#C4A8E0",
    "Light Red": "#E8737A", "Light Turquoise": "#8FDCD8", "Light Yellow": "#F7EBA0",
    "Off White": "#F5F1E8", "Orange": "#E8731F", "Other Blue": "#3A78C2",
    "Other Orange": "#E0833A", "Other Pink": "#D9669A", "Other Red": "#C0262D",
    "Pink": "#E57FA0", "Purple": "#7B3FA0", "Red": "#C8102E",
    "Turquoise": "#1FB5B0", "White": "#FFFFFF", "Yellow": "#F2CD2B",
    "Yellowish Brown": "#8B6B3E",
    "Other": None, "Unknown": None,
}

# Traduccion exacta por colour_group_name (evita discrepancias donde el dataset
# tenia Dark Red etiquetado como "lila" o Black como "desconocido").
COLOR_NOMBRE_ES: dict[str, str] = {
    "Beige": "beige",
    "Black": "negro",
    "Blue": "azul",
    "Dark Beige": "camel",
    "Dark Blue": "azul marino",
    "Dark Green": "verde oscuro",
    "Dark Grey": "gris oscuro",
    "Dark Orange": "terracota",
    "Dark Pink": "frambuesa",
    "Dark Purple": "morado oscuro",
    "Dark Red": "vino",
    "Dark Turquoise": "petroleo",
    "Dark Yellow": "mostaza",
    "Green": "verde",
    "Greenish Khaki": "verde caqui",
    "Grey": "gris",
    "Greyish Beige": "topo",
    "Light Beige": "beige claro",
    "Light Blue": "azul claro",
    "Light Green": "verde menta",
    "Light Grey": "gris claro",
    "Light Orange": "durazno",
    "Light Pink": "rosa claro",
    "Light Purple": "lila",
    "Light Red": "coral",
    "Light Turquoise": "turquesa claro",
    "Light Yellow": "amarillo claro",
    "Off White": "blanco hueso",
    "Orange": "naranja",
    "Other Blue": "azul",
    "Other Orange": "naranja",
    "Other Pink": "rosa",
    "Other Red": "rojo",
    "Pink": "rosa",
    "Purple": "morado",
    "Red": "rojo",
    "Turquoise": "turquesa",
    "White": "blanco",
    "Yellow": "amarillo",
    "Yellowish Brown": "cafe",
}

# Traduccion de respaldo del color "maestro" (perceived_colour_master_name).
COLOR_ES = {
    "Blue": "azul", "Khaki green": "verde caqui", "Grey": "gris",
    "Yellow": "amarillo", "Beige": "beige", "White": "blanco",
    "Orange": "naranja", "Black": "negro", "Brown": "cafe", "Green": "verde",
    "Red": "rojo", "Mole": "topo", "Pink": "rosa", "Lilac Purple": "lila",
    "Turquoise": "turquesa", "Unknown": "desconocido", "undefined": "desconocido",
}

# Colores disponibles para que el usuario elija una preferencia.
COLORES_PREFERIDOS = [
    "ninguno", "negro", "blanco", "gris", "beige", "cafe", "azul",
    "verde", "rojo", "vino", "rosa", "naranja", "amarillo", "morado", "lila",
    "turquesa", "topo",
]

_HEX_FAMILIA = {
    "negro": "#111111", "blanco": "#FFFFFF", "gris": "#808080",
    "beige": "#D8C3A5", "cafe": "#8B6B3E", "azul": "#2E5DA8",
    "verde": "#2E8B57", "rojo": "#C8102E", "vino": "#7A1420",
    "rosa": "#E57FA0", "naranja": "#E8731F", "amarillo": "#F2CD2B",
    "morado": "#7B3FA0", "lila": "#C4A8E0", "turquesa": "#1FB5B0",
    "topo": "#A38A63",
}


def nombre_color_es(color_nombre: Optional[str], color_es_actual: Optional[str] = None) -> str:
    """Devuelve un nombre en espanol coherente con el color HEX real de la prenda."""
    if isinstance(color_nombre, str):
        limpio = color_nombre.strip()
        if limpio in COLOR_NOMBRE_ES:
            return COLOR_NOMBRE_ES[limpio]
    if isinstance(color_es_actual, str) and color_es_actual.strip():
        return color_es_actual.strip()
    return "desconocido"


# --------------------------------------------------------------------------
# 2) HEX -> RGB -> CIELAB + HSL
# --------------------------------------------------------------------------
def hex_a_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_a_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    def lin(c: int) -> float:
        v = c / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b)
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def rgb_a_hsl(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Devuelve (hue 0..360, sat 0..1, lightness 0..1) en rueda cromatica estandar."""
    r, g, b = (c / 255.0 for c in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return (h * 360.0) % 360.0, s, l


@dataclass(frozen=True)
class ColorInfo:
    hex: str
    L: float          # luminosidad CIELAB 0 (negro) .. 100 (blanco)
    a: float          # eje verde (-) .. rojo (+)
    b: float          # eje azul (-) .. amarillo (+)
    C: float          # croma CIELAB: 0 = acromatico, alto = saturado
    h: float          # matiz HSL en grados 0..360 (rueda de color real)
    sat: float        # saturacion HSL 0..1
    temp: str         # "calido" | "frio" | "neutro"
    familia: str      # familia cromatica real (azul, rojo, vino, beige, etc.)
    es_neutro: bool   # True si funciona como neutro/comodín al combinar ropa


def _clasificar_familia_y_temp(
    L: float, C: float, hue_hsl: float, sat: float
) -> tuple[str, str, bool]:
    """Clasifica familia cromatica, temperatura y neutralidad usando HSL + Lab."""
    # 1. Acromaticos puros (negro, blanco, grises)
    if C < 5.5 or sat < 0.07:
        if L < 22:
            return "negro", "neutro", True
        if L > 90:
            return "blanco", "neutro", True
        return "gris", "neutro", True

    # 2. Blanco hueso / crema muy claro
    if L > 91 and C < 12:
        return "blanco", "calido", True

    # 3. Neutros tierra apagados (topo, beige, camel, cafe)
    if 22 <= hue_hsl <= 52 and (C < 26 or sat < 0.42 or (L < 55 and sat < 0.48)):
        if (C < 12 or sat < 0.18) and L >= 55:
            return "topo", "neutro", True
        if L < 52:
            return "cafe", "calido", True
        return "beige", "calido", True

    # 4. Verde caqui / oliva (entre amarillo-verdoso apagado)
    if 52 < hue_hsl <= 85 and (C < 30 or sat < 0.45) and L < 65:
        return "verde_caqui", "calido", False

    # 5. Familias cromaticas por matiz HSL real
    if hue_hsl < 12 or hue_hsl >= 346:
        if L < 33:
            return "vino", "frio", False
        if L > 58 and C > 38 and (hue_hsl < 12 or hue_hsl >= 354):
            return "coral", "calido", False
        if L > 75 and sat < 0.72:
            return "rosa", "frio", False
        return "rojo", "calido", False

    if 12 <= hue_hsl < 40:
        if L < 50 and sat < 0.50:
            return "cafe", "calido", True
        return "naranja", "calido", False

    if 40 <= hue_hsl < 70:
        return "amarillo", "calido", False

    if 70 <= hue_hsl < 165:
        temp = "calido" if hue_hsl < 115 else "frio"
        return "verde", temp, False

    if 165 <= hue_hsl < 198:
        return "turquesa", "frio", False

    if 198 <= hue_hsl < 255:
        # El azul marino oscuro funciona como base neutra al combinar outfits,
        # aunque frente a la piel mantiene temperatura fria.
        es_marino = L < 25 and sat < 0.60
        return "azul", "frio", es_marino

    if 255 <= hue_hsl < 305:
        if L >= 65:
            return "lila", "frio", False
        return "morado", "frio", False

    # 305 <= hue_hsl < 346: rosas, magentas y frambuesa
    return "rosa", "frio", False


def info_color(hex_color: Optional[str]) -> Optional[ColorInfo]:
    if not hex_color or not isinstance(hex_color, str):
        return None
    hex_clean = hex_color.strip()
    if not hex_clean:
        return None
    rgb = hex_a_rgb(hex_clean)
    L, a, b = rgb_a_lab(rgb)
    C = math.hypot(a, b)
    hue_hsl, sat, _ = rgb_a_hsl(rgb)
    familia, temp, es_neutro = _clasificar_familia_y_temp(L, C, hue_hsl, sat)
    return ColorInfo(
        hex=hex_clean.upper(),
        L=L, a=a, b=b, C=C,
        h=hue_hsl, sat=sat,
        temp=temp, familia=familia, es_neutro=es_neutro,
    )


def familia_color(color: Optional[ColorInfo]) -> str:
    """Familia cromatica de la prenda (para reglas de armonia y variedad)."""
    if color is None:
        return "neutro"
    if color.familia in {"negro", "blanco", "gris", "beige", "topo", "cafe"}:
        return "neutro"
    return color.familia


# --------------------------------------------------------------------------
# 3) Perfil de piel: profundidad + subtono + estacion cromatica
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PerfilPiel:
    hex: str
    L: float
    a: float
    b: float
    profundidad: str   # "clara" | "media" | "oscura"
    subtono: str       # "calido" | "frio" | "neutro"
    estacion: str      # "primavera" | "verano" | "otono" | "invierno"


def perfil_piel(hex_piel: str) -> PerfilPiel:
    rgb = hex_a_rgb(hex_piel)
    L, a, b = rgb_a_lab(rgb)
    C = math.hypot(a, b)
    h_lab = math.degrees(math.atan2(b, a)) % 360

    profundidad = "clara" if L >= 74 else "media" if L >= 48 else "oscura"

    # En piel humana (espacio Lab):
    # - h_lab < 53 (mayor proporcion de a* rosado respecto a b* dorado) -> frio
    # - h_lab > 61 (mayor proporcion de b* dorado/oliva) -> calido
    # - 53..61 o croma muy bajo -> neutro equilibrado
    if C < 5.5:
        subtono = "neutro"
    elif h_lab < 53.0:
        subtono = "frio"
    elif h_lab > 61.0:
        subtono = "calido"
    else:
        subtono = "neutro"

    if subtono == "calido":
        estacion = "primavera" if profundidad == "clara" else "otono"
    elif subtono == "frio":
        estacion = "verano" if profundidad == "clara" else "invierno"
    else:
        estacion = "verano" if profundidad == "clara" else (
            "otono" if b >= a * 1.6 else "invierno"
        )

    return PerfilPiel(hex_piel.upper(), L, a, b, profundidad, subtono, estacion)


# --------------------------------------------------------------------------
# 4) Puntaje de compatibilidad Color de Prenda x Tono de Piel (0..1)
# --------------------------------------------------------------------------
_TEMP_PIEL_PRENDA = {
    ("calido", "calido"): 1.00, ("calido", "neutro"): 0.86, ("calido", "frio"): 0.52,
    ("frio", "frio"): 1.00,     ("frio", "neutro"): 0.88,   ("frio", "calido"): 0.50,
    ("neutro", "calido"): 0.90, ("neutro", "neutro"): 0.92, ("neutro", "frio"): 0.90,
}

# Ajustes finos por familia cromatica segun (profundidad, subtono)
# Basado en colorimetria personal: evita que las pieles medias pierdan tonos joya/tierra
# y que las pieles oscuras solo reciban beige/blanco en lugar de tonos ricos o luminosos.
_AFINIDAD_FAMILIA_PIEL: dict[tuple[str, str], dict[str, float]] = {
    ("clara", "frio"): {
        "azul": 1.00, "vino": 0.98, "rosa": 0.94, "morado": 0.95, "lila": 0.92,
        "verde": 0.94, "turquesa": 0.94, "gris": 0.92, "negro": 0.90, "blanco": 0.92,
        "rojo": 0.80, "coral": 0.72, "beige": 0.55, "topo": 0.78, "cafe": 0.62,
        "verde_caqui": 0.60, "naranja": 0.42, "amarillo": 0.45,
    },
    ("clara", "calido"): {
        "coral": 1.00, "turquesa": 0.96, "verde": 0.94, "verde_caqui": 0.92,
        "rojo": 0.94, "naranja": 0.90, "amarillo": 0.86, "cafe": 0.90, "azul": 0.86,
        "vino": 0.86, "blanco": 0.88, "beige": 0.68, "topo": 0.80, "rosa": 0.72,
        "negro": 0.76, "gris": 0.74, "morado": 0.62, "lila": 0.58,
    },
    ("clara", "neutro"): {
        "azul": 0.96, "verde": 0.95, "turquesa": 0.96, "vino": 0.96, "rosa": 0.90,
        "coral": 0.90, "rojo": 0.90, "topo": 0.86, "gris": 0.88, "negro": 0.88,
        "blanco": 0.90, "verde_caqui": 0.84, "morado": 0.88, "lila": 0.85,
        "cafe": 0.82, "beige": 0.62, "naranja": 0.72, "amarillo": 0.70,
    },
    ("media", "calido"): {
        "verde_caqui": 1.00, "naranja": 0.96, "coral": 0.96, "rojo": 0.95,
        "amarillo": 0.94, "verde": 0.95, "turquesa": 0.96, "vino": 0.92,
        "blanco": 0.94, "azul": 0.88, "cafe": 0.86, "beige": 0.82, "negro": 0.84,
        "topo": 0.78, "gris": 0.75, "rosa": 0.70, "morado": 0.68, "lila": 0.56,
    },
    ("media", "frio"): {
        "azul": 1.00, "vino": 0.98, "verde": 0.96, "turquesa": 0.96, "rosa": 0.94,
        "morado": 0.95, "lila": 0.90, "blanco": 0.94, "gris": 0.90, "negro": 0.92,
        "rojo": 0.86, "topo": 0.82, "beige": 0.72, "coral": 0.74,
        "verde_caqui": 0.66, "cafe": 0.65, "amarillo": 0.55, "naranja": 0.48,
    },
    ("media", "neutro"): {
        "turquesa": 0.98, "verde": 0.96, "azul": 0.96, "vino": 0.96, "rojo": 0.92,
        "coral": 0.92, "blanco": 0.94, "verde_caqui": 0.90, "rosa": 0.88,
        "morado": 0.88, "negro": 0.88, "gris": 0.86, "topo": 0.82, "cafe": 0.82,
        "beige": 0.78, "lila": 0.82, "amarillo": 0.82, "naranja": 0.82,
    },
    ("oscura", "calido"): {
        "amarillo": 1.00, "naranja": 0.98, "coral": 0.98, "rojo": 0.96,
        "blanco": 0.96, "verde": 0.96, "turquesa": 0.96, "beige": 0.92,
        "verde_caqui": 0.90, "vino": 0.92, "azul": 0.90, "rosa": 0.88,
        "morado": 0.86, "lila": 0.82, "negro": 0.82, "topo": 0.80,
        "gris": 0.76, "cafe": 0.62,
    },
    ("oscura", "frio"): {
        "blanco": 0.98, "azul": 0.98, "rosa": 0.96, "morado": 0.96, "lila": 0.94,
        "turquesa": 0.96, "verde": 0.95, "vino": 0.94, "rojo": 0.92,
        "amarillo": 0.88, "gris": 0.86, "beige": 0.86, "negro": 0.84,
        "coral": 0.84, "topo": 0.78, "verde_caqui": 0.75, "naranja": 0.74, "cafe": 0.58,
    },
    ("oscura", "neutro"): {
        "blanco": 0.98, "turquesa": 0.98, "verde": 0.96, "amarillo": 0.96,
        "rojo": 0.96, "coral": 0.95, "azul": 0.95, "rosa": 0.94, "vino": 0.93,
        "naranja": 0.93, "morado": 0.92, "lila": 0.90, "beige": 0.90,
        "verde_caqui": 0.86, "negro": 0.84, "gris": 0.82, "topo": 0.80, "cafe": 0.60,
    },
}


def delta_e_lab(c: ColorInfo, piel: PerfilPiel) -> float:
    """Distancia perceptual CIE76 entre el color de la prenda y la piel."""
    return math.sqrt((c.L - piel.L) ** 2 + (c.a - piel.a) ** 2 + (c.b - piel.b) ** 2)


def puntaje_color_piel(
    color: Optional[ColorInfo],
    piel: PerfilPiel,
    cerca_rostro: bool = True,
) -> float:
    """Que tan bien le queda el color de la prenda al tono de piel del usuario.

    Combina:
      1. Subtono (calido/frio/neutro) y afinidad de familia cromatica con la
         profundidad de piel (clara/media/oscura).
      2. Prevencion del efecto 'apagado/camuflaje' cuando el color de una prenda
         cercana al rostro es demasiado parecido al propio color de la piel
         (Delta E bajo y poca diferencia de luminosidad).
      3. Si la prenda va lejos del rostro (pantalon/falda/calzado), los neutros
         versatiles no se castigan por subtono facial.
    """
    if color is None:
        return 0.58

    if not cerca_rostro:
        # En parte inferior o calzado, los neutros (negro, marino, beige, gris,
        # blanco, cafe, topo) son pilares del guardarropa sin importar el subtono facial.
        if color.es_neutro or color.familia == "verde_caqui":
            return 0.92
        temp_s = _TEMP_PIEL_PRENDA[(piel.subtono, color.temp)]
        fam_s = _AFINIDAD_FAMILIA_PIEL[(piel.profundidad, piel.subtono)].get(color.familia, 0.80)
        return round(0.35 * temp_s + 0.45 * fam_s + 0.20 * 0.85, 4)

    temp_s = _TEMP_PIEL_PRENDA[(piel.subtono, color.temp)]
    fam_s = _AFINIDAD_FAMILIA_PIEL[(piel.profundidad, piel.subtono)].get(color.familia, 0.78)

    # Evaluacion de contraste saludable frente al rostro (evitar efecto nude/apagado)
    de = delta_e_lab(color, piel)
    dl = abs(color.L - piel.L)
    if de < 16 or (dl < 10 and color.C < 28):
        # Muy similar al tono de piel: apaga el rostro
        contraste_s = 0.30
    elif de < 24 and dl < 16 and color.familia in {"beige", "topo", "cafe", "rosa", "naranja"}:
        contraste_s = 0.52
    elif dl >= 18 or color.C >= 32:
        contraste_s = 0.98
    else:
        contraste_s = 0.82

    return round(0.35 * temp_s + 0.45 * fam_s + 0.20 * contraste_s, 4)


# --------------------------------------------------------------------------
# 5) Armonia entre colores de prendas (pares y outfit completo)
# --------------------------------------------------------------------------
# Pares cromaticos especificos (simetricos): (fam1, fam2) -> puntaje
_PARES_CROMATICOS: dict[tuple[str, str], float] = {
    # Combinaciones clasicas de alto estilismo
    ("azul", "vino"): 0.94,
    ("azul", "verde_caqui"): 0.94,
    ("azul", "amarillo"): 0.90,
    ("azul", "naranja"): 0.88,
    ("azul", "coral"): 0.90,
    ("azul", "rosa"): 0.88,
    ("azul", "turquesa"): 0.90,
    ("azul", "rojo"): 0.88,
    ("azul", "verde"): 0.84,
    ("azul", "lila"): 0.86,
    ("azul", "morado"): 0.82,
    ("verde_caqui", "vino"): 0.93,
    ("verde_caqui", "naranja"): 0.92,
    ("verde_caqui", "amarillo"): 0.90,
    ("verde_caqui", "coral"): 0.88,
    ("verde_caqui", "verde"): 0.86,
    ("verde_caqui", "rosa"): 0.82,
    ("vino", "rosa"): 0.90,
    ("vino", "verde"): 0.88,
    ("vino", "amarillo"): 0.86,
    ("vino", "coral"): 0.82,
    ("vino", "lila"): 0.80,
    ("verde", "turquesa"): 0.88,
    ("verde", "amarillo"): 0.84,
    ("morado", "lila"): 0.90,
    ("morado", "rosa"): 0.84,
    ("morado", "turquesa"): 0.84,
    ("lila", "rosa"): 0.88,
    ("lila", "turquesa"): 0.86,
    ("coral", "turquesa"): 0.90,
    # Choques cromaticos clasicos que se deben evitar
    ("rojo", "amarillo"): 0.38,
    ("rojo", "naranja"): 0.42,
    ("rojo", "morado"): 0.38,
    ("rojo", "lila"): 0.42,
    ("rojo", "rosa"): 0.48,
    ("rojo", "verde"): 0.44,
    ("rojo", "coral"): 0.48,
    ("naranja", "rosa"): 0.36,
    ("naranja", "morado"): 0.35,
    ("naranja", "lila"): 0.38,
    ("naranja", "vino"): 0.52,
    ("amarillo", "rosa"): 0.46,
    ("amarillo", "lila"): 0.50,
    ("amarillo", "morado"): 0.52,
    ("amarillo", "naranja"): 0.46,
    ("amarillo", "coral"): 0.48,
    ("amarillo", "rojo"): 0.38,
    ("verde", "morado"): 0.46,
    ("verde", "rosa"): 0.50,
    ("verde", "naranja"): 0.54,
}


def armonia_colores(c1: Optional[ColorInfo], c2: Optional[ColorInfo]) -> float:
    """Que tan bien combinan dos prendas entre si (0..1)."""
    if c1 is None or c2 is None:
        return 0.68

    f1, f2 = c1.familia, c2.familia
    dl = abs(c1.L - c2.L)

    # Caso 1: Ambas prendas actuan como neutros (incluyendo azul marino oscuro)
    if c1.es_neutro and c2.es_neutro:
        if c1.hex == c2.hex:
            # Negro + negro estiliza bien; blanco + blanco o beige identico luce plano
            return 0.86 if f1 == "negro" else 0.68
        # Negro + cafe muy oscuro sin contraste puede verse confuso
        if {f1, f2} == {"negro", "cafe"} and dl < 35:
            return 0.74
        # Dos tonos de la misma familia neutra con buen contraste tonal (ej. beige claro + camel)
        if f1 == f2:
            return 0.92 if dl >= 15 else 0.74
        return 0.96

    # Caso 2: Una prenda neutra + una prenda con color
    if c1.es_neutro or c2.es_neutro:
        neutro = c1 if c1.es_neutro else c2
        crom = c2 if c1.es_neutro else c1
        # Neutros tierra calidos (cafe, beige) chocan ligeramente con pasteles frios intensos
        if neutro.familia in {"cafe", "beige"} and crom.familia in {"lila", "morado", "rosa"} and crom.C > 35:
            return 0.76
        # Gris frio con naranja/amarillo chillante baja un poco salvo contraste alto
        if neutro.familia == "gris" and crom.familia in {"naranja", "amarillo"} and dl < 18:
            return 0.78
        return 0.96 if dl >= 14 else 0.89

    # Caso 3: Dos prendas cromaticas de la misma familia (look monocromatico / tonal)
    if f1 == f2:
        if dl >= 20:
            return 0.88   # ej. camisa azul claro + pantalon azul intenso
        if dl >= 10:
            return 0.76
        return 0.60       # exactamente el mismo color chillante arriba y abajo

    # Caso 4: Dos familias cromaticas distintas
    par = _PARES_CROMATICOS.get((f1, f2), _PARES_CROMATICOS.get((f2, f1)))
    if par is not None:
        # Si ambos son hiper saturados a la vez (C > 55), moderamos ligeramente
        if c1.C > 55 and c2.C > 55 and par > 0.6:
            return round(par - 0.08, 4)
        return par

    # Respaldo por distancia en rueda HSL
    dh = abs(c1.h - c2.h)
    dh = min(dh, 360.0 - dh)
    if dh <= 40:
        return 0.82 if dl >= 12 else 0.70
    if 150 <= dh <= 210:
        return 0.84 if (c1.C < 55 or c2.C < 55) else 0.72
    return 0.65


def armonia_outfit_colores(
    piezas_colores: dict[str, Optional[ColorInfo]],
) -> tuple[float, list[str]]:
    """Evalua la paleta completa del outfit (regla de 3 colores, base neutra y adyacencia)."""
    razones: list[str] = []
    presentes = {k: c for k, c in piezas_colores.items() if c is not None}
    if len(presentes) < 2:
        return 0.85, razones

    # Pares con pesos segun cercania visual real en el cuerpo
    pesos_pares = [
        (("top", "bottom"), 0.38),
        (("top", "abrigo"), 0.24),
        (("vestido", "abrigo"), 0.42),
        (("bottom", "abrigo"), 0.14),
        (("bottom", "calzado"), 0.14),
        (("vestido", "calzado"), 0.38),
        (("top", "calzado"), 0.10),
        (("abrigo", "calzado"), 0.10),
    ]
    suma_p, total_w, min_adyacente = 0.0, 0.0, 1.0
    for (k1, k2), w in pesos_pares:
        if k1 in presentes and k2 in presentes:
            s = armonia_colores(presentes[k1], presentes[k2])
            suma_p += s * w
            total_w += w
            if (k1, k2) in {("top", "bottom"), ("top", "abrigo"), ("vestido", "abrigo")}:
                min_adyacente = min(min_adyacente, s)

    puntaje = suma_p / total_w if total_w > 0 else 0.85

    # No permitir que unos zapatos neutros escondan un choque directo entre piezas superiores
    if min_adyacente < 0.62:
        puntaje *= 0.72

    # Regla estilistica de paleta global (base neutra + 1 o 2 acentos vs exceso de colores)
    cromaticos = [c.familia for c in presentes.values() if not c.es_neutro]
    neutros = [c.familia for c in presentes.values() if c.es_neutro]
    familias_crom_unicas = set(cromaticos)

    if len(familias_crom_unicas) == 0:
        # Look minimalista de neutros: excelente si hay contraste de luminosidad
        luces = [c.L for c in presentes.values()]
        if max(luces) - min(luces) >= 22:
            puntaje = min(0.98, puntaje + 0.02)
            razones.append("paleta neutra equilibrada con buen contraste de luz")
        else:
            puntaje *= 0.88
    elif len(familias_crom_unicas) == 1 and len(neutros) >= 1:
        # Regla de oro: base neutra + 1 color protagonista
        if len(cromaticos) == 1:
            puntaje = min(0.99, puntaje + 0.03)
            razones.append("base neutra combinada con un color protagonista armonioso")
        else:
            razones.append("acento de color coordinado sobre base neutra")
    elif len(familias_crom_unicas) == 2 and min_adyacente >= 0.84:
        # Si hay 2 familias cromaticas pero mas de 2 piezas con color (ej. top, zapatos y abrigo), moderar
        if len(cromaticos) >= 3:
            puntaje *= 0.86
        else:
            razones.append("combinacion cromatica equilibrada entre prendas")
    elif len(familias_crom_unicas) >= 3:
        # 3 o mas colores no neutros distintos suelen saturar el outfit
        puntaje *= 0.65 if len(familias_crom_unicas) == 3 else 0.50

    # Evitar repetir 3+ veces la misma familia de color en todo el conjunto (ej. triple azul o triple beige)
    conteo_familias: dict[str, int] = {}
    for c in presentes.values():
        conteo_familias[c.familia] = conteo_familias.get(c.familia, 0) + 1
    if max(conteo_familias.values(), default=0) >= 3:
        puntaje *= 0.78

    # Evitar bloque identico arriba y abajo (ej. camisa blanca + pantalon blanco puro)
    if "top" in presentes and "bottom" in presentes:
        ct, cb = presentes["top"], presentes["bottom"]
        if ct.hex == cb.hex and ct.familia != "negro":
            puntaje *= 0.76
        elif not ct.es_neutro and ct.familia == cb.familia and abs(ct.L - cb.L) < 12:
            puntaje *= 0.78

    # Coordinacion del calzado con el resto del outfit
    calz = presentes.get("calzado")
    if calz is not None and not calz.es_neutro:
        otras_fams = {c.familia for k, c in presentes.items() if k != "calzado"}
        otras_crom = {c.familia for k, c in presentes.items() if k != "calzado" and not c.es_neutro}
        # Calzado de color llamativo funciona si hace eco de una prenda o si todo lo demas es neutro
        if calz.familia not in otras_fams and len(otras_crom) >= 1:
            puntaje *= 0.82

    return round(max(0.0, min(1.0, puntaje)), 4), razones


def puntaje_preferencia_color(
    color: Optional[ColorInfo],
    preferencia: Optional[str],
) -> float:
    """Mide cuanto se acerca una prenda al color preferido del usuario."""
    preferencia = str(preferencia).strip().lower() if preferencia else ""
    if not preferencia or preferencia == "ninguno" or color is None:
        return 0.5

    if color.familia == preferencia:
        return 1.0
    if preferencia == "rojo" and color.familia == "coral":
        return 0.86
    if preferencia == "rojo" and color.familia == "vino":
        return 0.78
    if preferencia == "vino" and color.familia == "rojo" and color.L < 45:
        return 0.85
    if preferencia == "verde" and color.familia == "verde_caqui":
        return 0.85
    if preferencia == "morado" and color.familia == "lila":
        return 0.86
    if preferencia == "lila" and color.familia == "morado":
        return 0.86
    if preferencia == "beige" and color.familia == "topo":
        return 0.86
    if preferencia == "topo" and color.familia in {"beige", "cafe"}:
        return 0.84

    objetivo = _HEX_FAMILIA.get(preferencia)
    if objetivo is None:
        return 0.5

    objetivo_info = info_color(objetivo)
    if objetivo_info is None:
        return 0.5

    L_diff = abs(color.L - objetivo_info.L) / 100.0
    hue_diff = abs(color.h - objetivo_info.h)
    hue_diff = min(hue_diff, 360.0 - hue_diff) / 180.0

    if preferencia in {"negro", "blanco", "gris", "beige", "cafe", "topo"}:
        if not color.es_neutro:
            return 0.45
        return round(max(0.45, 0.92 - 0.65 * L_diff), 4)
    return round(max(0.40, 0.90 - 0.75 * hue_diff - 0.25 * L_diff), 4)
