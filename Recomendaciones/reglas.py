"""Reglas de estilismo (conocimiento de dominio) como tablas editables.

Cada tabla da un puntaje 0..1: que tan bien le va un CORTE a una FORMA de
cuerpo, y un AJUSTE a un SOMATOTIPO. Son reglas de estilismo comunes; son
tu punto de partida y se pueden afinar sin tocar el resto del codigo.
Las llaves de forma son EXACTAMENTE las que devuelve tu clasificador
(Formas_Cuerpo/src/body_shape_classifier.py), normalizadas con norm().
"""
from __future__ import annotations

import unicodedata


def norm(texto) -> str:
    """minusculas, sin tildes, sin espacios sobrantes."""
    if texto is None:
        return ""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.strip().lower()


# --------------------------------------------------------------------------
# FORMA DE CUERPO x CORTE
# cortes del dataset: entallado, linea_a, peplum, oversize, cruzado, sin_corte
# --------------------------------------------------------------------------
CORTE_POR_FORMA = {
    "Mujer": {
        "reloj de arena":      {"entallado": 1.0, "cruzado": 1.0, "peplum": 0.7, "linea_a": 0.7, "oversize": 0.4},
        "triangulo / pera":    {"linea_a": 1.0, "cruzado": 0.8, "entallado": 0.7, "oversize": 0.5, "peplum": 0.5},
        "triangulo invertido": {"linea_a": 1.0, "peplum": 0.9, "cruzado": 0.6, "entallado": 0.6, "oversize": 0.4},
        "rectangulo":          {"peplum": 1.0, "cruzado": 0.9, "entallado": 0.8, "linea_a": 0.7, "oversize": 0.6},
        "ovalado / manzana":   {"cruzado": 1.0, "linea_a": 0.9, "oversize": 0.7, "entallado": 0.4, "peplum": 0.3},
    },
    "Hombre": {
        "trapecio":            {"entallado": 0.9, "linea_a": 0.8, "oversize": 0.7, "cruzado": 0.5, "peplum": 0.3},
        "triangulo invertido": {"linea_a": 0.9, "oversize": 0.8, "entallado": 0.7, "cruzado": 0.5, "peplum": 0.3},
        "rectangulo":          {"oversize": 0.9, "linea_a": 0.8, "entallado": 0.5, "cruzado": 0.5, "peplum": 0.3},
        "triangulo":           {"oversize": 0.8, "linea_a": 0.7, "entallado": 0.4, "cruzado": 0.5, "peplum": 0.3},
        "ovalo":               {"linea_a": 0.9, "oversize": 0.6, "entallado": 0.3, "cruzado": 0.5, "peplum": 0.3},
    },
}

# --------------------------------------------------------------------------
# SOMATOTIPO x AJUSTE   (ajustes del dataset: holgado, regular, ajustado)
# --------------------------------------------------------------------------
AJUSTE_POR_SOMATOTIPO = {
    "ectomorfo": {"holgado": 1.0, "regular": 0.8, "ajustado": 0.4},   # delgado: sumar volumen
    "mesomorfo": {"ajustado": 1.0, "regular": 0.9, "holgado": 0.6},   # atletico: marcar silueta
    "endomorfo": {"regular": 1.0, "holgado": 0.7, "ajustado": 0.3},   # mas volumen: ni pegado ni bolsudo
}

# --------------------------------------------------------------------------
# COMPATIBILIDAD ENTRE ESTILOS (para armar outfits coherentes)
# --------------------------------------------------------------------------
_ESTILOS = {
    ("casual", "casual"): 1.0, ("casual", "deportivo"): 0.8, ("casual", "streetwear"): 0.8,
    ("casual", "elegante"): 0.6, ("casual", "formal"): 0.4,
    ("elegante", "elegante"): 1.0, ("elegante", "formal"): 0.9,
    ("elegante", "deportivo"): 0.2, ("elegante", "streetwear"): 0.3,
    ("formal", "formal"): 1.0, ("formal", "deportivo"): 0.1, ("formal", "streetwear"): 0.2,
    ("deportivo", "deportivo"): 1.0, ("deportivo", "streetwear"): 0.9,
    ("streetwear", "streetwear"): 1.0,
}


def compat_estilo(e1: str, e2: str) -> float:
    return _ESTILOS.get((e1, e2), _ESTILOS.get((e2, e1), 0.5))


# --------------------------------------------------------------------------
# PESOS DEL PUNTAJE (se pueden reemplazar con pesos.json entrenado)
# --------------------------------------------------------------------------
PESOS_DEFECTO = {"forma": 0.35, "somato": 0.20, "color": 0.30, "estilo": 0.15}

# El calzado casi no depende de la forma del cuerpo ni del somatotipo.
CATEGORIAS_SIN_CUERPO = {"calzado"}

# Peso de cada pieza dentro del outfit.
PESO_PIEZA = {"top": 0.35, "bottom": 0.35, "vestido": 0.70, "calzado": 0.15, "abrigo": 0.15}

# Valores de graphical_appearance_name que NO cuentan como estampado fuerte.
SIN_ESTAMPADO = {"Solid", "Melange", "Denim", "Treatment", "Other structure",
                 "Slub", "Chambray", "Unknown"}