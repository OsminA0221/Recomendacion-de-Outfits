"""PASO 2: motor de recomendacion.

Entrada : perfil de las 3 IAs (el mismo dict de Profile.to_dict()).
Salida  : prendas y outfits ordenados, con razones y puntajes.

Puntaje de una prenda (0..1):
    forma  * s_forma   (corte vs forma del cuerpo)
  + somato * s_somato  (ajuste vs somatotipo)
  + color  * s_color   (color vs piel)
  + estilo * s_estilo  (solo si el usuario pidio un estilo)
Los pesos viven en reglas.PESOS_DEFECTO o en pesos.json (entrenado).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .colores import (armonia_colores, info_color, perfil_piel,
                     puntaje_color_piel)
from .reglas import (AJUSTE_POR_SOMATOTIPO, CATEGORIAS_SIN_CUERPO,
                    CORTE_POR_FORMA, PESO_PIEZA, PESOS_DEFECTO,
                    SIN_ESTAMPADO, compat_estilo, norm)

AQUI = Path(__file__).resolve().parent      # carpeta Recomendaciones/
RAIZ = AQUI.parent                           # raiz del proyecto (donde esta data/)
DATA = RAIZ / "data"


@dataclass
class PerfilUsuario:
    genero: str            # "Mujer" | "Hombre"
    forma: str             # etiqueta del clasificador de forma
    somatotipo: str        # "ectomorfo" | "mesomorfo" | "endomorfo"
    somato_conf: float     # confianza del modelo YOLO (0..1)
    piel_hex: str          # HEX de la paleta de piel
    piel_label: str = ""


def perfil_desde_dict(d: dict) -> PerfilUsuario:
    """Convierte Profile.to_dict() (de Integracion/profile_store.py)."""
    genero = "Hombre" if norm(d.get("genero")) in ("hombre", "masculino", "h") else "Mujer"
    somato = norm(d["somatotipo"]["label"])
    for base in ("ecto", "meso", "endo"):          # tolera nombres con erratas
        if somato.startswith(base):
            somato = base + "morfo"
    return PerfilUsuario(
        genero=genero,
        forma=norm(d["forma_corporal"]["label"]),
        somatotipo=somato,
        somato_conf=float(d["somatotipo"].get("confidence", 1.0)),
        piel_hex=d["tono_piel"]["skin_tone"],
        piel_label=d["tono_piel"].get("label", ""),
    )


class Recomendador:
    def __init__(self, csv_path: Optional[Path] = None,
                 imagenes_dir: Optional[Path] = None,
                 pesos_path: Optional[Path] = None):
        self.csv_path = Path(csv_path or DATA / "prendas_limpio.csv")
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"No existe {self.csv_path}. Ejecuta primero: python -m Recomendaciones.preparar_dataset")
        self.imagenes_dir = Path(imagenes_dir or DATA / "Imagenes")
        self.df = pd.read_csv(self.csv_path)
        self.df["color_info"] = self.df["color_hex"].apply(info_color)
        self.df["estampado"] = ~self.df["patron"].isin(SIN_ESTAMPADO)

        self.pesos = dict(PESOS_DEFECTO)
        pesos_path = Path(pesos_path or AQUI / "pesos.json")
        if pesos_path.exists():
            self.pesos.update(json.loads(pesos_path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # puntajes individuales
    # ------------------------------------------------------------------
    @staticmethod
    def _s_forma(p: PerfilUsuario, corte: str) -> float:
        tabla = CORTE_POR_FORMA[p.genero].get(p.forma)
        if tabla is None:
            return 0.6                      # forma desconocida: neutro
        return tabla.get(corte, 0.6)        # "sin_corte" -> neutro

    @staticmethod
    def _s_somato(p: PerfilUsuario, ajuste: str) -> float:
        tabla = AJUSTE_POR_SOMATOTIPO.get(p.somatotipo)
        if tabla is None:
            return 0.6
        s = tabla.get(ajuste, 0.6)
        # Si el modelo duda, el puntaje se acerca a 0.5 (neutro)
        confianza = min(1.0, p.somato_conf / 0.6)
        return 0.5 + (s - 0.5) * confianza

    def puntuar(self, perfil: PerfilUsuario, estilo: Optional[str] = None) -> pd.DataFrame:
        """Devuelve las prendas del genero del usuario con todos los puntajes."""
        d = self.df[self.df["genero"] == perfil.genero].copy()
        piel = perfil_piel(perfil.piel_hex)

        d["s_forma"] = d["corte"].apply(lambda c: self._s_forma(perfil, c))
        d["s_somato"] = d["ajuste"].apply(lambda a: self._s_somato(perfil, a))
        d["s_color"] = d["color_info"].apply(lambda c: puntaje_color_piel(c, piel))
        if estilo:
            d["s_estilo"] = (d["estilo"] == estilo.lower()).astype(float)
        else:
            d["s_estilo"] = 0.5

        # pesos por fila (calzado ignora cuerpo; sin estilo se reparte el peso)
        pesos = pd.DataFrame(self.pesos, index=d.index)
        sin_cuerpo = d["categoria"].isin(CATEGORIAS_SIN_CUERPO)
        pesos.loc[sin_cuerpo, ["forma", "somato"]] = 0.0
        if not estilo:
            pesos["estilo"] = 0.0
        pesos = pesos.div(pesos.sum(axis=1), axis=0)

        d["score"] = (pesos["forma"] * d["s_forma"] + pesos["somato"] * d["s_somato"]
                      + pesos["color"] * d["s_color"] + pesos["estilo"] * d["s_estilo"])
        d.attrs["piel"] = piel
        return d.sort_values("score", ascending=False)

    # ------------------------------------------------------------------
    # recomendar prendas sueltas
    # ------------------------------------------------------------------
    def recomendar_prendas(self, perfil: PerfilUsuario, categoria: Optional[str] = None,
                           n: int = 10, estilo: Optional[str] = None) -> list[dict]:
        d = self.puntuar(perfil, estilo)
        if categoria:
            d = d[d["categoria"] == categoria]
        d = d.drop_duplicates("nombre").head(n)
        return [self._item_dict(r, perfil) for _, r in d.iterrows()]

    # ------------------------------------------------------------------
    # recomendar outfits completos
    # ------------------------------------------------------------------
    def recomendar_outfits(self, perfil: PerfilUsuario, n: int = 5,
                           estilo: Optional[str] = None, top_k: int = 10,
                           incluir_abrigo: bool = True) -> list[dict]:
        d = self.puntuar(perfil, estilo)

        def pool(cat):
            sub = d[d["categoria"] == cat].drop_duplicates("nombre").head(top_k)
            return [r for _, r in sub.iterrows()]

        tops, bottoms = pool("top"), pool("bottom")
        vestidos, calzados, abrigos = pool("vestido"), pool("calzado"), pool("abrigo")

        combos = []
        if tops and bottoms and calzados:
            combos += [{"top": t, "bottom": b, "calzado": c}
                       for t, b, c in itertools.product(tops, bottoms, calzados)]
        if vestidos and calzados:
            combos += [{"vestido": v, "calzado": c}
                       for v, c in itertools.product(vestidos, calzados)]

        evaluados = [(self._puntaje_outfit(c), c) for c in combos]
        evaluados.sort(key=lambda x: x[0], reverse=True)

        # Variedad: cuantas veces puede repetirse una misma pieza entre los outfits
        MAX_USO = {"top": 1, "vestido": 1, "bottom": 2, "calzado": 1, "abrigo": 1}
        elegidos, usos_nombre, usos_id = [], {}, {}

        def libre(cat, item):
            return usos_id.get(item["item_id"], 0) < MAX_USO[cat]

        for puntaje, piezas in evaluados:
            principal = piezas.get("vestido", piezas.get("top"))
            if usos_nombre.get(principal["nombre"], 0) >= 1:
                continue                      # no repetir la prenda principal
            if not all(libre(c, p) for c, p in piezas.items()):
                continue
            if incluir_abrigo and abrigos:
                piezas = dict(piezas)
                disponibles = [a for a in abrigos if libre("abrigo", a)]
                if disponibles:
                    piezas["abrigo"] = max(
                        disponibles, key=lambda a: a["score"] * self._compat_con(a, piezas))
                    puntaje = self._puntaje_outfit(piezas)
            elegidos.append((puntaje, piezas))
            usos_nombre[principal["nombre"]] = 1
            for c, p in piezas.items():
                usos_id[p["item_id"]] = usos_id.get(p["item_id"], 0) + 1
            if len(elegidos) == n:
                break

        elegidos.sort(key=lambda x: x[0], reverse=True)
        return [{
            "puntaje": round(float(pt), 3),
            "piezas": {k: self._item_dict(v, perfil) for k, v in pz.items()},
        } for pt, pz in elegidos]

    # ------------------------------------------------------------------
    # internos
    # ------------------------------------------------------------------
    @staticmethod
    def _compat_con(prenda, piezas: dict) -> float:
        vals = [0.5 * compat_estilo(prenda["estilo"], p["estilo"])
                + 0.5 * armonia_colores(prenda["color_info"], p["color_info"])
                for p in piezas.values()]
        return sum(vals) / len(vals)

    @staticmethod
    def _puntaje_outfit(piezas: dict) -> float:
        pesos = {k: PESO_PIEZA[k] for k in piezas}
        total = sum(pesos.values())
        base = sum(piezas[k]["score"] * w for k, w in pesos.items()) / total

        pares = list(itertools.combinations(piezas.values(), 2))
        est = sum(compat_estilo(a["estilo"], b["estilo"]) for a, b in pares) / len(pares)
        col = sum(armonia_colores(a["color_info"], b["color_info"]) for a, b in pares) / len(pares)

        n_estamp = sum(bool(p["estampado"]) for p in piezas.values())
        penal = {0: 1.0, 1: 1.0, 2: 0.85}.get(n_estamp, 0.7)   # 2+ estampados chocan
        return (0.6 * base + 0.2 * est + 0.2 * col) * penal

    def _ruta_imagen(self, p: str) -> str:
        return str(self.imagenes_dir / Path(str(p).replace("\\", "/")).name)

    def _item_dict(self, r, perfil: PerfilUsuario) -> dict:
        razones = []
        if r["categoria"] not in CATEGORIAS_SIN_CUERPO:
            if r["s_forma"] >= 0.85:
                razones.append(f"el corte {r['corte']} favorece tu forma ({perfil.forma})")
            if r["s_somato"] >= 0.85:
                razones.append(f"el ajuste {r['ajuste']} va bien con tu somatotipo ({perfil.somatotipo})")
        if r["s_color"] >= 0.8:
            razones.append(f"el color {r['color_es']} favorece tu tono de piel")
        return {
            "item_id": int(r["item_id"]), "nombre": r["nombre"],
            "categoria": r["categoria"], "subcategoria": r["subcategoria"],
            "estilo": r["estilo"], "corte": r["corte"], "ajuste": r["ajuste"],
            "color": r["color_es"], "color_hex": r["color_hex"],
            "imagen": self._ruta_imagen(r["imagen_path"]),
            "score": round(float(r["score"]), 3),
            "scores": {k: round(float(r[k]), 3)
                       for k in ("s_forma", "s_somato", "s_color", "s_estilo")},
            "razones": razones,
        }