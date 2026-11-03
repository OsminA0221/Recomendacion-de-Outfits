"""Reglas de estilismo (conocimiento de dominio) editables y explicables.

Define:
1. Compatibilidad entre FORMA CORPORAL, TIPO DE PRENDA (categoria/subcategoria) y CORTE.
2. Compatibilidad entre SOMATOTIPO, AJUSTE y pistas reales del nombre de la prenda.
3. Coherencia al COMBINAR PRENDAS en un outfit:
   - Compatibilidad de estilos y niveles de formalidad.
   - Coherencia climatica / estacional (no mezclar abrigo o sueter de lana con chanclas o shorts).
   - Reglas de capas (evitar sobrecapas incompatibles o calzado de casa como pantuflas).
   - Equilibrio de silueta y ubicacion estrategica del color segun el tipo de cuerpo.
"""
from __future__ import annotations

from functools import lru_cache
import unicodedata
from typing import Optional


@lru_cache(maxsize=4096)
def norm(texto) -> str:
    """Minusculas, sin tildes, sin espacios sobrantes."""
    if texto is None:
        return ""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.strip().lower()


# --------------------------------------------------------------------------
# 1) FORMA DE CUERPO x CORTE (general y diferenciado por categoria)
#    cortes del dataset: entallado, linea_a, peplum, oversize, cruzado, sin_corte
# --------------------------------------------------------------------------
CORTE_POR_FORMA = {
    "Mujer": {
        "reloj de arena":      {"entallado": 1.0, "cruzado": 1.0, "peplum": 0.85, "linea_a": 0.80, "sin_corte": 0.70, "oversize": 0.38},
        "triangulo / pera":    {"linea_a": 1.0, "cruzado": 0.90, "peplum": 0.80, "entallado": 0.70, "sin_corte": 0.75, "oversize": 0.60},
        "triangulo invertido": {"linea_a": 1.0, "peplum": 0.92, "cruzado": 0.88, "entallado": 0.72, "sin_corte": 0.75, "oversize": 0.45},
        "rectangulo":          {"peplum": 1.0, "cruzado": 0.95, "linea_a": 0.88, "entallado": 0.82, "sin_corte": 0.70, "oversize": 0.65},
        "ovalado / manzana":   {"cruzado": 1.0, "linea_a": 0.95, "sin_corte": 0.80, "oversize": 0.70, "entallado": 0.42, "peplum": 0.32},
    },
    "Hombre": {
        "trapecio":            {"entallado": 0.95, "sin_corte": 0.88, "linea_a": 0.82, "oversize": 0.70, "cruzado": 0.60, "peplum": 0.35},
        "triangulo invertido": {"sin_corte": 0.90, "linea_a": 0.88, "entallado": 0.78, "oversize": 0.75, "cruzado": 0.55, "peplum": 0.35},
        "rectangulo":          {"oversize": 0.88, "sin_corte": 0.85, "entallado": 0.80, "linea_a": 0.80, "cruzado": 0.55, "peplum": 0.35},
        "triangulo":           {"sin_corte": 0.90, "linea_a": 0.85, "oversize": 0.75, "entallado": 0.55, "cruzado": 0.55, "peplum": 0.35},
        "ovalo":               {"sin_corte": 0.90, "linea_a": 0.90, "oversize": 0.76, "entallado": 0.45, "cruzado": 0.55, "peplum": 0.35},
    },
}

# Reglas diferenciadas entre parte superior (top/abrigo) e inferior (bottom/vestido),
# porque un cuerpo pera o triangulo invertido necesita cortes opuestos arriba y abajo.
CORTE_POR_FORMA_CATEGORIA: dict[str, dict[str, dict[str, dict[str, float]]]] = {
    "Mujer": {
        "reloj de arena": {
            "top":     {"cruzado": 1.0, "entallado": 1.0, "peplum": 0.88, "sin_corte": 0.68, "linea_a": 0.68, "oversize": 0.35},
            "bottom":  {"entallado": 0.95, "linea_a": 0.92, "cruzado": 0.90, "sin_corte": 0.75, "peplum": 0.70, "oversize": 0.45},
            "vestido": {"cruzado": 1.0, "entallado": 1.0, "linea_a": 0.86, "peplum": 0.82, "sin_corte": 0.65, "oversize": 0.35},
            "abrigo":  {"entallado": 1.0, "cruzado": 1.0, "linea_a": 0.82, "sin_corte": 0.75, "oversize": 0.48},
        },
        "triangulo / pera": {
            # Estructurar/destacar hombros y cintura arriba; fluidez sin apretar cadera abajo
            "top":     {"peplum": 0.96, "cruzado": 0.96, "entallado": 0.85, "oversize": 0.78, "sin_corte": 0.78, "linea_a": 0.65},
            "bottom":  {"linea_a": 1.0, "cruzado": 0.88, "sin_corte": 0.80, "oversize": 0.74, "entallado": 0.52, "peplum": 0.42},
            "vestido": {"linea_a": 1.0, "cruzado": 0.95, "peplum": 0.68, "sin_corte": 0.68, "oversize": 0.55, "entallado": 0.48},
            "abrigo":  {"entallado": 0.95, "linea_a": 0.92, "sin_corte": 0.82, "oversize": 0.68},
        },
        "triangulo invertido": {
            # Limpiar volumen en hombros arriba; sumar volumen/vuelo en cadera abajo
            "top":     {"cruzado": 1.0, "entallado": 0.92, "linea_a": 0.86, "peplum": 0.88, "sin_corte": 0.80, "oversize": 0.35},
            "bottom":  {"linea_a": 1.0, "peplum": 0.96, "oversize": 0.92, "cruzado": 0.86, "sin_corte": 0.78, "entallado": 0.46},
            "vestido": {"linea_a": 1.0, "peplum": 0.96, "cruzado": 0.88, "sin_corte": 0.68, "entallado": 0.55, "oversize": 0.45},
            "abrigo":  {"linea_a": 0.96, "sin_corte": 0.82, "entallado": 0.80, "oversize": 0.42},
        },
        "rectangulo": {
            "top":     {"peplum": 1.0, "cruzado": 0.96, "entallado": 0.85, "linea_a": 0.78, "oversize": 0.68, "sin_corte": 0.68},
            "bottom":  {"linea_a": 0.96, "peplum": 0.92, "oversize": 0.82, "entallado": 0.80, "sin_corte": 0.74},
            "vestido": {"cruzado": 1.0, "peplum": 1.0, "linea_a": 0.92, "entallado": 0.78, "sin_corte": 0.65, "oversize": 0.50},
            "abrigo":  {"entallado": 0.95, "linea_a": 0.88, "sin_corte": 0.78, "oversize": 0.68},
        },
        "ovalado / manzana": {
            # Caida fluida en abdomen y piernas estilizadas abajo
            "top":     {"cruzado": 1.0, "linea_a": 0.96, "sin_corte": 0.80, "oversize": 0.74, "entallado": 0.36, "peplum": 0.30},
            "bottom":  {"linea_a": 0.95, "sin_corte": 0.88, "entallado": 0.84, "oversize": 0.68, "peplum": 0.38},
            "vestido": {"cruzado": 1.0, "linea_a": 0.96, "sin_corte": 0.74, "oversize": 0.68, "entallado": 0.32, "peplum": 0.30},
            "abrigo":  {"linea_a": 0.95, "sin_corte": 0.88, "entallado": 0.80, "oversize": 0.72},
        },
    },
    "Hombre": {
        "trapecio": {
            "top":    {"entallado": 0.96, "sin_corte": 0.88, "linea_a": 0.82, "oversize": 0.68, "cruzado": 0.60, "peplum": 0.40},
            "bottom": {"entallado": 0.95, "sin_corte": 0.88, "linea_a": 0.82, "oversize": 0.68},
            "abrigo": {"entallado": 0.96, "sin_corte": 0.88, "oversize": 0.76, "linea_a": 0.78},
        },
        "triangulo invertido": {
            "top":    {"sin_corte": 0.92, "entallado": 0.86, "linea_a": 0.85, "oversize": 0.64, "cruzado": 0.60, "peplum": 0.42},
            "bottom": {"oversize": 0.92, "sin_corte": 0.90, "linea_a": 0.88, "entallado": 0.56},
            "abrigo": {"sin_corte": 0.88, "entallado": 0.86, "linea_a": 0.82, "oversize": 0.66},
        },
        "rectangulo": {
            "top":    {"oversize": 0.90, "sin_corte": 0.84, "entallado": 0.78, "linea_a": 0.80, "cruzado": 0.62, "peplum": 0.45},
            "bottom": {"sin_corte": 0.88, "entallado": 0.86, "linea_a": 0.82, "oversize": 0.78},
            "abrigo": {"entallado": 0.92, "oversize": 0.88, "sin_corte": 0.86, "linea_a": 0.82},
        },
        "triangulo": {
            "top":    {"sin_corte": 0.88, "linea_a": 0.86, "oversize": 0.76, "entallado": 0.46, "cruzado": 0.58, "peplum": 0.40},
            "bottom": {"sin_corte": 0.92, "linea_a": 0.86, "oversize": 0.72, "entallado": 0.72},
            "abrigo": {"entallado": 0.92, "sin_corte": 0.92, "linea_a": 0.86, "oversize": 0.74},
        },
        "ovalo": {
            "top":    {"sin_corte": 0.90, "linea_a": 0.92, "oversize": 0.78, "entallado": 0.38, "cruzado": 0.58, "peplum": 0.35},
            "bottom": {"sin_corte": 0.92, "linea_a": 0.88, "oversize": 0.78, "entallado": 0.66},
            "abrigo": {"sin_corte": 0.94, "linea_a": 0.90, "oversize": 0.82, "entallado": 0.70},
        },
    },
}

# --------------------------------------------------------------------------
# 2) AFINIDAD DE TIPO DE PRENDA (subcategoria) x FORMA DE CUERPO
# --------------------------------------------------------------------------
SUBCATEGORIA_POR_FORMA: dict[str, dict[str, dict[str, float]]] = {
    "Mujer": {
        "reloj de arena": {
            "dress": 0.98, "jumpsuit/playsuit": 0.94, "bodysuit": 0.95, "blouse": 0.94,
            "top": 0.92, "shirt": 0.88, "skirt": 0.96, "trousers": 0.92, "blazer": 0.95,
            "coat": 0.92, "jacket": 0.88, "cardigan": 0.86, "t-shirt": 0.84,
            "vest top": 0.86, "sweater": 0.80, "shorts": 0.82, "hoodie": 0.58,
        },
        "triangulo / pera": {
            "blouse": 0.98, "shirt": 0.94, "top": 0.90, "blazer": 0.98, "jacket": 0.92,
            "coat": 0.90, "dress": 0.96, "skirt": 0.96, "trousers": 0.88, "cardigan": 0.88,
            "sweater": 0.86, "t-shirt": 0.84, "vest top": 0.80, "jumpsuit/playsuit": 0.82,
            "bodysuit": 0.82, "shorts": 0.58, "hoodie": 0.70,
        },
        "triangulo invertido": {
            "skirt": 0.98, "trousers": 0.94, "dress": 0.92, "top": 0.92, "t-shirt": 0.92,
            "vest top": 0.90, "cardigan": 0.94, "blouse": 0.86, "shirt": 0.84,
            "shorts": 0.88, "jumpsuit/playsuit": 0.85, "coat": 0.86, "jacket": 0.80,
            "blazer": 0.68, "sweater": 0.78, "bodysuit": 0.80, "hoodie": 0.62,
        },
        "rectangulo": {
            "blouse": 0.96, "dress": 0.95, "skirt": 0.96, "jumpsuit/playsuit": 0.92,
            "top": 0.92, "blazer": 0.94, "jacket": 0.92, "cardigan": 0.90, "shirt": 0.88,
            "trousers": 0.88, "sweater": 0.88, "bodysuit": 0.86, "coat": 0.88,
            "shorts": 0.84, "t-shirt": 0.82, "vest top": 0.82, "hoodie": 0.74,
        },
        "ovalado / manzana": {
            "blouse": 0.98, "cardigan": 0.96, "dress": 0.94, "shirt": 0.92, "blazer": 0.96,
            "coat": 0.94, "trousers": 0.92, "skirt": 0.86, "top": 0.86, "jacket": 0.86,
            "t-shirt": 0.82, "sweater": 0.80, "vest top": 0.62, "shorts": 0.64,
            "jumpsuit/playsuit": 0.58, "hoodie": 0.64, "bodysuit": 0.34,
        },
    },
    "Hombre": {
        "trapecio": {
            "shirt": 0.96, "polo shirt": 0.96, "t-shirt": 0.92, "sweater": 0.92,
            "trousers": 0.96, "blazer": 0.96, "jacket": 0.94, "coat": 0.92,
            "cardigan": 0.88, "tailored waistcoat": 0.90, "hoodie": 0.80,
            "shorts": 0.80, "top": 0.78, "vest top": 0.62,
        },
        "triangulo invertido": {
            "polo shirt": 0.96, "t-shirt": 0.94, "shirt": 0.92, "sweater": 0.88,
            "trousers": 0.96, "shorts": 0.86, "cardigan": 0.90, "jacket": 0.86,
            "coat": 0.86, "blazer": 0.82, "hoodie": 0.80, "top": 0.80,
            "tailored waistcoat": 0.75, "vest top": 0.65,
        },
        "rectangulo": {
            "sweater": 0.96, "shirt": 0.94, "hoodie": 0.90, "cardigan": 0.94,
            "polo shirt": 0.90, "jacket": 0.96, "blazer": 0.94, "coat": 0.92,
            "trousers": 0.94, "t-shirt": 0.84, "shorts": 0.80,
            "tailored waistcoat": 0.86, "top": 0.76, "vest top": 0.52,
        },
        "triangulo": {
            "shirt": 0.96, "polo shirt": 0.92, "blazer": 0.98, "jacket": 0.95,
            "coat": 0.92, "cardigan": 0.90, "sweater": 0.88, "trousers": 0.96,
            "t-shirt": 0.82, "hoodie": 0.74, "shorts": 0.62,
            "tailored waistcoat": 0.78, "top": 0.68, "vest top": 0.35,
        },
        "ovalo": {
            "shirt": 0.98, "polo shirt": 0.94, "cardigan": 0.94, "blazer": 0.98,
            "coat": 0.94, "jacket": 0.90, "trousers": 0.98, "sweater": 0.84,
            "t-shirt": 0.82, "hoodie": 0.68, "tailored waistcoat": 0.72,
            "shorts": 0.56, "top": 0.62, "vest top": 0.30,
        },
    },
}

# Calidad/versatilidad base de calzado y prendas especiales para outfits de calle
CALIDAD_CALZADO: dict[str, float] = {
    "sneakers": 0.94,
    "boots": 0.92,
    "other shoe": 0.92,
    "pumps": 0.94,
    "heeled sandals": 0.90,
    "flat shoe": 0.92,
    "flat shoes": 0.92,
    "ballerinas": 0.90,
    "wedge": 0.86,
    "sandals": 0.80,
    "flip flop": 0.20,   # solo sube en ocasion == "playa"
    "slippers": 0.08,    # pantuflas de casa: nunca recomendar para salir
}


def puntaje_forma_prenda(
    genero: str,
    forma: str,
    categoria: str,
    subcategoria: str,
    corte: str,
    nombre: str = "",
    estilo: Optional[str] = None,
    ocasion: Optional[str] = None,
) -> float:
    """Combina compatibilidad de CORTE + TIPO DE PRENDA (subcategoria) con la forma corporal."""
    sub = norm(subcategoria)
    cat = norm(categoria)
    oc = norm(ocasion)
    est = norm(estilo)
    nom = norm(nombre)

    # Descartar disfraces y controlar prendas exclusivas de deporte o playa/casa
    if sub == "costumes":
        return 0.05
    if cat == "calzado":
        if sub == "flip flop" and oc == "playa":
            return 0.95
        if sub == "sandals" and oc == "playa":
            return 0.98
        return CALIDAD_CALZADO.get(sub, 0.82)

    if sub == "leggings/tights" and est != "deportivo" and oc != "deportivo":
        return 0.25

    tabla_cat = CORTE_POR_FORMA_CATEGORIA.get(genero, {}).get(forma, {}).get(cat)
    if tabla_cat is not None:
        s_corte = tabla_cat.get(corte, 0.68)
    else:
        tabla_gen = CORTE_POR_FORMA.get(genero, {}).get(forma)
        s_corte = tabla_gen.get(corte, 0.65) if tabla_gen else 0.65

    s_sub = SUBCATEGORIA_POR_FORMA.get(genero, {}).get(forma, {}).get(sub, 0.78)

    # Pistas del nombre real del producto (corrige etiquetas ruidosas del dataset)
    ajuste_nombre = 0.0
    if any(k in nom for k in ("super skinny", "skinny", "jegging", "muscle")):
        if forma in {"ovalo", "ovalado / manzana", "triangulo", "triangulo / pera", "triangulo invertido"} and cat in {"top", "bottom"}:
            # En pera/ovalo/triangulo invertido, skinny extremo o muscle fit rompe la proporcion
            if not (forma in {"ovalado / manzana"} and cat == "bottom" and "super skinny" not in nom):
                ajuste_nombre -= 0.16
    if any(k in nom for k in ("relaxed", "wide", "loose")) and cat == "bottom":
        if forma == "triangulo invertido":
            ajuste_nombre += 0.06

    return round(max(0.05, min(1.0, 0.60 * s_corte + 0.40 * s_sub + ajuste_nombre)), 4)


# --------------------------------------------------------------------------
# 3) SOMATOTIPO x AJUSTE   (ajustes del dataset: holgado, regular, ajustado)
# --------------------------------------------------------------------------
AJUSTE_POR_SOMATOTIPO = {
    "ectomorfo": {"holgado": 0.95, "regular": 0.90, "ajustado": 0.42},   # delgado: estructura y capas sin pegarse
    "mesomorfo": {"regular": 0.96, "ajustado": 0.95, "holgado": 0.65},   # atletico: definir silueta con naturalidad
    "endomorfo": {"regular": 1.00, "holgado": 0.76, "ajustado": 0.25},   # mayor volumen: caida limpia, ni apretado ni saco
}


def puntaje_somato_prenda(
    somatotipo: str,
    somato_conf: float,
    categoria: str,
    subcategoria: str,
    ajuste: str,
    nombre: str = "",
) -> float:
    """Evalua que tan bien va el ajuste y estructura de la prenda con el somatotipo."""
    if norm(categoria) in CATEGORIAS_SIN_CUERPO:
        return 0.85
    tabla = AJUSTE_POR_SOMATOTIPO.get(somatotipo)
    if tabla is None:
        return 0.65
    s = tabla.get(ajuste, 0.65)
    nom = norm(nombre)
    sub = norm(subcategoria)

    # Refinar con el nombre real de la prenda
    if somatotipo == "endomorfo":
        if any(k in nom for k in ("super skinny", "skinny", "jegging", "muscle", "body")):
            s -= 0.22
        elif sub in {"shirt", "blazer", "cardigan", "coat", "trousers"} and ajuste == "regular":
            s = min(1.0, s + 0.04)
    elif somatotipo == "ectomorfo":
        if any(k in nom for k in ("super skinny", "muscle")):
            s -= 0.18
        elif sub in {"sweater", "hoodie", "jacket", "blazer", "cardigan", "shirt"}:
            s = min(1.0, s + 0.05)

    s = max(0.15, min(1.0, s))
    confianza = min(1.0, max(0.35, somato_conf / 0.6))
    return round(0.5 + (s - 0.5) * confianza, 4)


# --------------------------------------------------------------------------
# 4) COMPATIBILIDAD ENTRE PRENDAS (estilo + clima + tipo de prenda)
# --------------------------------------------------------------------------
_ESTILOS = {
    ("casual", "casual"): 1.0, ("casual", "deportivo"): 0.78, ("casual", "streetwear"): 0.86,
    ("casual", "elegante"): 0.68, ("casual", "formal"): 0.52,
    ("elegante", "elegante"): 1.0, ("elegante", "formal"): 0.92,
    ("elegante", "deportivo"): 0.18, ("elegante", "streetwear"): 0.28,
    ("formal", "formal"): 1.0, ("formal", "deportivo"): 0.10, ("formal", "streetwear"): 0.18,
    ("deportivo", "deportivo"): 1.0, ("deportivo", "streetwear"): 0.88,
    ("streetwear", "streetwear"): 1.0,
}


def compat_estilo(e1: str, e2: str) -> float:
    return _ESTILOS.get((e1, e2), _ESTILOS.get((e2, e1), 0.5))


_SUB_FRESCAS = {"shorts", "vest top", "sandals", "flip flop", "wedge"}
_SUB_INVIERNO = {"sweater", "hoodie", "coat", "boots"}
_PALABRAS_INVIERNO = ("wool", "padded", "parka", "pile", "raincoat", "trench", "fleece", "knit", "down")
_PALABRAS_VERANO = ("linen", "beach", "flip flop", "espadril")


@lru_cache(maxsize=2048)
def nivel_termico(subcategoria: str, nombre: str = "", categoria: str = "") -> str:
    """Clasifica una prenda en 'fresco', 'templado' o 'abrigo_frio'."""
    sub = norm(subcategoria)
    nom = norm(nombre)
    if sub in _SUB_FRESCAS or any(k in nom for k in ("flip flop", "espadril")):
        return "fresco"
    if sub in _SUB_INVIERNO or any(k in nom for k in _PALABRAS_INVIERNO):
        return "abrigo_frio"
    if "linen" in nom and norm(categoria) in {"top", "bottom"}:
        return "fresco"
    return "templado"


_SUB_CAPA_EXTERIOR = {"sweater", "hoodie", "cardigan", "blazer", "coat", "jacket", "tailored waistcoat"}
_NOM_CAPA_EXTERIOR = (
    "jacket", "coat", "blazer", "cardigan", "sweater", "hoodie", "shacket",
    "fleece", "parka", "puffer", "bomber", "jumper", "pullover", "sweatshirt",
)


def es_capa_exterior(prenda: dict) -> bool:
    """True si la prenda es un sueter/chaqueta/abrigo (capa superior que abriga),
    sin importar si el dataset la clasifico como 'top' o como 'abrigo'."""
    if norm(prenda.get("categoria")) == "abrigo":
        return True
    if norm(prenda.get("subcategoria")) in _SUB_CAPA_EXTERIOR:
        return True
    nombre = norm(prenda.get("nombre"))
    return any(k in nombre for k in _NOM_CAPA_EXTERIOR)


def compat_prendas(p1: dict, p2: dict, ocasion: Optional[str] = None) -> float:
    """Evalua si dos prendas tienen sentido juntas por estilo, clima y funcion."""
    s_est = compat_estilo(norm(p1.get("estilo")), norm(p2.get("estilo")))
    sub1, sub2 = norm(p1.get("subcategoria")), norm(p2.get("subcategoria"))
    cat1, cat2 = norm(p1.get("categoria")), norm(p2.get("categoria"))
    nom1, nom2 = norm(p1.get("nombre")), norm(p2.get("nombre"))
    subs = {sub1, sub2}
    cats = {cat1, cat2}

    penal = 1.0

    # 0. Nunca dos sueteres/chaquetas/abrigos en el mismo outfit
    if (
        cat1 in {"top", "abrigo"} and cat2 in {"top", "abrigo"}
        and es_capa_exterior(p1) and es_capa_exterior(p2)
    ):
        return 0.05

    # 1. Nunca mezclar pantuflas de casa o disfraces
    if "slippers" in subs or "costumes" in subs:
        return 0.05
    if "flip flop" in subs and norm(ocasion) != "playa":
        penal *= 0.25

    # 2. Coherencia climatica (ej. no usar sueter de lana / abrigo pesado con shorts o sandalias)
    t1 = nivel_termico(sub1, nom1, cat1)
    t2 = nivel_termico(sub2, nom2, cat2)
    if {t1, t2} == {"fresco", "abrigo_frio"}:
        # Excepcion estilistica femenina: vestido/falda con botas funciona muy bien;
        # pero shorts/sandalias/tank-top con abrigo/sueter/botas choca.
        if not ("boots" in subs and subs & {"dress", "skirt"}):
            penal *= 0.30

    # 3. Coherencia de formalidad por subcategoria
    formales = {"blazer", "tailored waistcoat", "coat"}
    muy_informales = {"shorts", "flip flop", "sandals", "hoodie", "vest top", "leggings/tights", "dungarees"}
    if (subs & formales) and (subs & muy_informales):
        # Unicamente streetwear permite hoodie + blazer/coat de forma intencional
        if not (norm(ocasion) == "streetwear" and subs <= {"hoodie", "blazer", "coat"}):
            penal *= 0.38

    # Chaleco de vestir (tailored waistcoat) requiere camisa/pantalon formal y zapato cerrado
    if "tailored waistcoat" in subs and (subs & {"sneakers", "t-shirt", "shorts", "sandals", "jacket"}):
        penal *= 0.40

    # Pants deportivos (sweatpants/joggers) solo combinan bien con prendas deportivas/streetwear
    if any(k in nom1 or k in nom2 for k in ("sweatpant", "jogger")):
        if subs & {"blazer", "coat", "shirt", "blouse", "pumps", "heeled sandals", "other shoe"}:
            penal *= 0.42
        elif "sweater" in subs and norm(ocasion) not in {"deportivo", "streetwear"}:
            penal *= 0.72

    # 4. Evitar doble capa redundante o torpe entre top y abrigo
    if cats == {"top", "abrigo"}:
        top_p = p1 if cat1 == "top" else p2
        abr_p = p2 if cat1 == "top" else p1
        top_sub, abr_sub = norm(top_p.get("subcategoria")), norm(abr_p.get("subcategoria"))
        top_nom = norm(top_p.get("nombre"))
        if "shacket" in top_nom:
            penal *= 0.35   # un shacket ya es sobrecamisa/chaqueta
        elif top_sub == "cardigan" and abr_sub in {"blazer", "jacket"}:
            penal *= 0.55
        elif top_sub == "hoodie" and abr_sub == "blazer":
            penal *= 0.50

    # 5. Vestidos elegantes no van con chaquetas deportivas ni tenis de gimnasio
    if "dress" in subs:
        dress_p = p1 if sub1 == "dress" else p2
        otra_p = p2 if sub1 == "dress" else p1
        if norm(dress_p.get("estilo")) in {"elegante", "formal"}:
            if norm(otra_p.get("estilo")) == "deportivo" or norm(otra_p.get("subcategoria")) == "sneakers":
                penal *= 0.42

    return round(max(0.05, min(1.0, s_est * penal)), 4)


def equilibrio_cuerpo_outfit(
    piezas: dict,
    genero: str,
    forma: str,
    somatotipo: str,
    estilo: Optional[str] = None,
) -> tuple[float, list[str]]:
    """Evalua en escala 0..1 como trabajan JUNTAS las piezas para favorecer la silueta."""
    razones: list[str] = []
    puntaje = 0.84
    est = norm(estilo)

    top = piezas.get("top")
    bottom = piezas.get("bottom")
    vestido = piezas.get("vestido")
    abrigo = piezas.get("abrigo")

    # 1. Equilibrio de volumenes (evitar todo holgado/oversize o todo ultra pegado)
    prendas_cuerpo = [p for k, p in piezas.items() if k in {"top", "bottom", "vestido", "abrigo"}]
    n_holgadas = sum(
        p.get("corte") == "oversize" or p.get("ajuste") == "holgado"
        for p in prendas_cuerpo
    )
    if len(prendas_cuerpo) >= 3 and n_holgadas >= 3:
        puntaje -= 0.08 if est == "streetwear" else 0.22
    elif top is not None and bottom is not None:
        top_vol = top.get("corte") == "oversize" or top.get("ajuste") == "holgado"
        bot_vol = bottom.get("corte") in {"oversize", "linea_a"} or bottom.get("ajuste") == "holgado"
        top_aj = top.get("corte") in {"entallado", "cruzado"} or top.get("ajuste") == "ajustado"
        bot_aj = bottom.get("corte") == "entallado" or bottom.get("ajuste") == "ajustado"

        if top_vol and bot_vol and est != "streetwear":
            puntaje -= 0.14   # exceso de tela arriba y abajo pierde definicion
        elif top_aj and bot_aj and somatotipo == "endomorfo":
            puntaje -= 0.22   # todo ajustado no favorece somatotipo endomorfo
        elif (top_aj and bot_vol) or (top_vol and not bot_vol) or (
            top.get("ajuste") == "regular" and bottom.get("ajuste") == "regular"
        ):
            puntaje += 0.06
            razones.append("proporcion equilibrada de volumen entre parte superior e inferior")

    # 2. Ubicacion estrategica del color y lineas segun la forma corporal
    ct = top.get("color_info") if top is not None else None
    cb = bottom.get("color_info") if bottom is not None else None

    if ct is not None and cb is not None:
        if forma in {"triangulo / pera", "triangulo"}:
            if ct.L >= cb.L + 12 or (not ct.es_neutro and cb.es_neutro and cb.L < 55):
                puntaje += 0.08
                razones.append(f"tono superior luminoso e inferior sobrio para equilibrar tu silueta ({forma})")
            elif cb.L >= ct.L + 22 and not cb.es_neutro:
                puntaje -= 0.12

        elif forma == "triangulo invertido":
            if cb.L >= ct.L + 10 or (ct.es_neutro and ct.L < 55 and (cb.L > 50 or not cb.es_neutro)):
                puntaje += 0.08
                razones.append("tono superior sobrio y mayor presencia inferior para armonizar hombros")
            elif ct.C > 45 and cb.L < 25 and bottom.get("corte") == "entallado":
                puntaje -= 0.14

        elif forma in {"ovalado / manzana", "ovalo"}:
            if abs(ct.L - cb.L) <= 22 and ct.L < 68:
                puntaje += 0.06
                razones.append("continuidad tonal que estiliza verticalmente el torso")
            elif ct.L < cb.L or ct.L < 55:
                puntaje += 0.05
                razones.append(f"parte superior sobria que favorece tu silueta ({forma})")
            if abrigo is not None and norm(abrigo.get("subcategoria")) in {"blazer", "coat", "jacket"}:
                puntaje += 0.04
                razones.append("tercera pieza estructurada que aporta lineas verticales")

        elif forma == "rectangulo":
            if abs(ct.L - cb.L) >= 18 or ct.familia != cb.familia:
                puntaje += 0.07
                razones.append("contraste entre parte superior e inferior para definir cintura")

        elif forma in {"reloj de arena", "trapecio"}:
            if top.get("corte") in {"entallado", "cruzado", "peplum"} or top.get("ajuste") == "regular":
                puntaje += 0.07
                razones.append(f"cortes que acompanan naturalmente tu proporcion ({forma})")

    elif vestido is not None:
        # Un vestido del corte adecuado es una pieza integral sumamente favorecedora
        if vestido.get("s_forma", 0.0) >= 0.88:
            puntaje += 0.10
            razones.append(f"pieza integral de corte {vestido.get('corte')} ideal para tu silueta ({forma})")
        if abrigo is not None and norm(abrigo.get("subcategoria")) in {"blazer", "coat"}:
            puntaje += 0.04
            razones.append("vestido complementado con capa estructurada")

    return round(max(0.40, min(1.0, puntaje)), 4), razones


def puntaje_ocasion(estilo: str, subcategoria: str, ocasion: Optional[str]) -> float:
    """Evalua si una prenda encaja en una ocasion elegida por el usuario."""
    if not ocasion:
        return 1.0
    estilo = norm(estilo)
    subcategoria = norm(subcategoria)

    if subcategoria in {"costumes", "slippers"}:
        return 0.05

    if ocasion == "casual":
        if subcategoria == "flip flop":
            return 0.25
        return 1.0 if estilo in {"casual", "streetwear"} else 0.72
    if ocasion in {"elegante", "formal"}:
        if subcategoria in {"shorts", "flip flop", "hoodie", "vest top", "leggings/tights", "sneakers"}:
            return 0.15
        return 1.0 if estilo in {"elegante", "formal"} else 0.42
    if ocasion == "deportivo":
        if subcategoria in {"blazer", "coat", "pumps", "heeled sandals", "dress", "skirt"}:
            return 0.15
        return 1.0 if estilo == "deportivo" else 0.35
    if ocasion == "playa":
        playa = {"shorts", "skirt", "dress", "top", "vest top", "sandals",
                 "flip flop", "wedge", "blouse", "t-shirt", "shirt"}
        invierno = {"coat", "sweater", "hoodie", "boots", "blazer", "jacket"}
        if subcategoria in invierno:
            return 0.10
        return 1.0 if estilo == "casual" and subcategoria in playa else (
            0.72 if subcategoria in playa else 0.25)
    if ocasion == "frio":
        frio = {"coat", "jacket", "blazer", "cardigan", "sweater", "hoodie",
                "trousers", "boots"}
        verano = {"shorts", "flip flop", "sandals", "vest top"}
        if subcategoria in verano:
            return 0.10
        return 1.0 if subcategoria in frio else (
            0.68 if estilo in {"casual", "elegante", "formal"} else 0.35)
    if ocasion == "streetwear":
        return 1.0 if estilo == "streetwear" else (
            0.80 if subcategoria in {"hoodie", "sneakers", "t-shirt", "jacket", "trousers"} else 0.45
        )
    return 0.5


# --------------------------------------------------------------------------
# 5) PESOS DEL PUNTAJE (se pueden reemplazar con pesos.json)
# --------------------------------------------------------------------------
PESOS_DEFECTO = {"forma": 0.35, "somato": 0.20, "color": 0.30, "estilo": 0.15}

# El calzado no depende de forma corporal ni somatotipo, pero si de calidad/funcion y estilo.
CATEGORIAS_SIN_CUERPO = {"calzado"}

# Categorias cercanas al rostro donde el subtono del color impacta directamente la piel.
CATEGORIAS_CERCA_ROSTRO = {"top", "vestido", "abrigo"}

# Peso de cada pieza dentro del outfit.
PESO_PIEZA = {"top": 0.36, "bottom": 0.34, "vestido": 0.70, "calzado": 0.15, "abrigo": 0.15}

# Valores de graphical_appearance_name que NO cuentan como estampado fuerte.
SIN_ESTAMPADO = {"Solid", "Melange", "Denim", "Treatment", "Other structure",
                 "Slub", "Chambray", "Unknown"}
