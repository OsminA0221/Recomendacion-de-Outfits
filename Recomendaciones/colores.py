"""Utilidades de color.

- COLOR_HEX: da un codigo HEX a cada nombre de color del dataset (el dataset
  solo trae nombres como "Dark Blue"; la deteccion de piel entrega HEX).
- Conversion HEX -> CIELAB (espacio donde la distancia se parece a como ve
  el ojo humano) y datos derivados: luminosidad, croma, matiz, temperatura.
- Perfil de piel (profundidad + subtono) y puntajes de color.
"""
from __future__ import annotations

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

# Traduccion del color "maestro" (perceived_colour_master_name) para mostrar.
COLOR_ES = {
    "Blue": "azul", "Khaki green": "verde caqui", "Grey": "gris",
    "Yellow": "amarillo", "Beige": "beige", "White": "blanco",
    "Orange": "naranja", "Black": "negro", "Brown": "cafe", "Green": "verde",
    "Red": "rojo", "Mole": "topo", "Pink": "rosa", "Lilac Purple": "lila",
    "Turquoise": "turquesa", "Unknown": "desconocido", "undefined": "desconocido",
}


# --------------------------------------------------------------------------
# 2) HEX -> RGB -> CIELAB
# --------------------------------------------------------------------------
def hex_a_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_a_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


@dataclass(frozen=True)
class ColorInfo:
    hex: str
    L: float          # luminosidad 0 (negro) .. 100 (blanco)
    C: float          # croma: 0 = gris, alto = color intenso
    h: float          # matiz en grados 0..360
    temp: str         # "calido" | "frio" | "neutro"


def info_color(hex_color: Optional[str]) -> Optional[ColorInfo]:
    if not hex_color or not isinstance(hex_color, str):
        return None
    L, a, b = rgb_a_lab(hex_a_rgb(hex_color))
    C = math.hypot(a, b)
    h = math.degrees(math.atan2(b, a)) % 360
    if C < 20:                      # grises, negro, blanco, beiges apagados
        temp = "neutro"
    elif 15 <= h < 110:             # rojos, naranjas, amarillos, caquis
        temp = "calido"
    else:                           # verdes, azules, morados, rosas frios
        temp = "frio"
    return ColorInfo(hex_color.upper(), L, C, h, temp)


# --------------------------------------------------------------------------
# 3) Perfil de piel: profundidad + subtono
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PerfilPiel:
    hex: str
    L: float
    profundidad: str   # "clara" | "media" | "oscura"
    subtono: str       # "calido" | "frio" | "neutro"


def perfil_piel(hex_piel: str) -> PerfilPiel:
    L, a, b = rgb_a_lab(hex_a_rgb(hex_piel))
    C = math.hypot(a, b)
    h = math.degrees(math.atan2(b, a)) % 360

    profundidad = "clara" if L >= 75 else "media" if L >= 50 else "oscura"

    if C < 8:            # croma muy bajo: el matiz no es confiable
        subtono = "neutro"
    elif h < 50:         # tira a rosado
        subtono = "frio"
    elif h > 68:         # tira a dorado/amarillo
        subtono = "calido"
    else:
        subtono = "neutro"
    return PerfilPiel(hex_piel.upper(), L, profundidad, subtono)


# --------------------------------------------------------------------------
# 4) Puntajes de color (0..1)
# --------------------------------------------------------------------------
_TEMP_PIEL_PRENDA = {
    # (subtono de piel, temperatura de prenda) -> puntaje
    ("calido", "calido"): 1.0, ("calido", "neutro"): 0.85, ("calido", "frio"): 0.55,
    ("frio", "frio"): 1.0,     ("frio", "neutro"): 0.85,   ("frio", "calido"): 0.55,
    ("neutro", "calido"): 0.85, ("neutro", "neutro"): 0.9, ("neutro", "frio"): 0.85,
}


def puntaje_color_piel(color: Optional[ColorInfo], piel: PerfilPiel) -> float:
    """Que tan bien le queda el color de la prenda a esa piel.

    Mezcla dos ideas clasicas de colorimetria:
      * temperatura: colores calidos con subtono calido, frios con frio.
      * contraste: la prenda debe separarse lo suficiente de la piel (en L).
    """
    if color is None:
        return 0.6  # color desconocido: ni premia ni castiga
    temp = _TEMP_PIEL_PRENDA[(piel.subtono, color.temp)]
    contraste = min(1.0, max(0.4, abs(color.L - piel.L) / 35.0))
    return round(0.55 * temp + 0.45 * contraste, 4)


def armonia_colores(c1: Optional[ColorInfo], c2: Optional[ColorInfo]) -> float:
    """Que tan bien combinan dos prendas entre si (0..1)."""
    if c1 is None or c2 is None:
        return 0.7
    if c1.temp == "neutro" and c2.temp == "neutro":
        return 1.0
    if c1.temp == "neutro" or c2.temp == "neutro":
        return 0.95   # un neutro combina con casi todo
    dh = abs(c1.h - c2.h)
    dh = min(dh, 360 - dh)
    if dh <= 30:
        return 1.0    # monocromatico
    if dh <= 70:
        return 0.85   # analogo
    if dh >= 150:
        return 0.7    # complementario (fuerte, pero valido)
    return 0.5        # zona que suele chocar