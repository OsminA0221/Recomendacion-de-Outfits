"""PASO 2: motor de recomendacion de prendas y combinacion inteligente de outfits.

Entrada : perfil de las 3 IAs (el mismo dict de Profile.to_dict()).
Salida  : prendas y outfits ordenados, con razones explicables y puntajes.

Evalua dos niveles complementarios:
1. Nivel prenda vs usuario:
   - Tipo de prenda (subcategoria) y corte vs forma corporal del usuario.
   - Ajuste y estructura vs somatotipo del usuario.
   - Colorimetria personal (profundidad + subtono de piel + cercania al rostro).
   - Estilo, ocasion y color preferido opcionales.
2. Nivel combinacion de outfit:
   - Coherencia funcional, formal y climatica entre tipos de prenda.
   - Armonia cromatica global (base neutra + acento, contraste tonal, calzado coordinado).
   - Equilibrio de volumen y ubicacion estrategica del color segun la forma corporal.
   - Control de estampados para evitar choques visuales.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .colores import (
    PerfilPiel,
    armonia_outfit_colores,
    info_color,
    nombre_color_es,
    perfil_piel,
    puntaje_color_piel,
    puntaje_preferencia_color,
)
from .reglas import (
    CATEGORIAS_CERCA_ROSTRO,
    CATEGORIAS_SIN_CUERPO,
    PESO_PIEZA,
    PESOS_DEFECTO,
    SIN_ESTAMPADO,
    compat_prendas,
    equilibrio_cuerpo_outfit,
    es_capa_exterior,
    nivel_termico,
    norm,
    puntaje_forma_prenda,
    puntaje_ocasion,
    puntaje_somato_prenda,
)

AQUI = Path(__file__).resolve().parent      # carpeta Recomendaciones/
RAIZ = AQUI.parent                           # raiz del proyecto (donde esta data/)
DATA = RAIZ / "data"

# Traduccion legible de subcategorias frecuentes para las explicaciones al usuario
_SUBCAT_ES = {
    "t-shirt": "camiseta", "shirt": "camisa", "polo shirt": "polo",
    "sweater": "sueter", "hoodie": "sudadera", "cardigan": "cardigan",
    "blouse": "blusa", "top": "top", "vest top": "top sin mangas",
    "bodysuit": "body", "tailored waistcoat": "chaleco de vestir",
    "trousers": "pantalon", "shorts": "shorts", "skirt": "falda",
    "leggings/tights": "leggings", "dress": "vestido",
    "jumpsuit/playsuit": "enterizo", "dungarees": "overol",
    "blazer": "blazer", "coat": "abrigo", "jacket": "chaqueta",
    "sneakers": "tenis", "boots": "botas", "other shoe": "zapato cerrado",
    "pumps": "tacones", "heeled sandals": "sandalias de tacon",
    "flat shoe": "zapato plano", "flat shoes": "zapato plano",
    "ballerinas": "balerinas", "sandals": "sandalias", "wedge": "plataformas",
    "flip flop": "sandalias de playa", "slippers": "calzado de descanso",
}


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
        self.df["color_es"] = [
            nombre_color_es(cn, ce)
            for cn, ce in zip(self.df["color_nombre"], self.df["color_es"])
        ]
        self.df["color_info"] = self.df["color_hex"].apply(info_color)
        self.df["estampado"] = ~self.df["patron"].isin(SIN_ESTAMPADO)

        self.pesos = dict(PESOS_DEFECTO)
        pesos_path = Path(pesos_path or AQUI / "pesos.json")
        if pesos_path.exists():
            self.pesos.update(json.loads(pesos_path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # puntajes individuales (prenda vs cuerpo y piel del usuario)
    # ------------------------------------------------------------------
    @staticmethod
    def _s_forma(
        p: PerfilUsuario,
        corte: str,
        categoria: str = "top",
        subcategoria: str = "",
        nombre: str = "",
        estilo: Optional[str] = None,
        ocasion: Optional[str] = None,
    ) -> float:
        return puntaje_forma_prenda(
            p.genero, p.forma, categoria, subcategoria, corte, nombre, estilo, ocasion
        )

    @staticmethod
    def _s_somato(
        p: PerfilUsuario,
        ajuste: str,
        categoria: str = "top",
        subcategoria: str = "",
        nombre: str = "",
    ) -> float:
        return puntaje_somato_prenda(
            p.somatotipo, p.somato_conf, categoria, subcategoria, ajuste, nombre
        )

    def puntuar(
        self,
        perfil: PerfilUsuario,
        estilo: Optional[str] = None,
        ocasion: Optional[str] = None,
        color_preferido: Optional[str] = None,
    ) -> pd.DataFrame:
        """Devuelve las prendas del genero del usuario con todos los puntajes."""
        d = self.df[self.df["genero"] == perfil.genero].copy()
        piel = perfil_piel(perfil.piel_hex)

        d["s_forma"] = [
            self._s_forma(perfil, crt, cat, sub, nom, estilo, ocasion)
            for crt, cat, sub, nom in zip(
                d["corte"], d["categoria"], d["subcategoria"], d["nombre"]
            )
        ]
        d["s_somato"] = [
            self._s_somato(perfil, aj, cat, sub, nom)
            for aj, cat, sub, nom in zip(
                d["ajuste"], d["categoria"], d["subcategoria"], d["nombre"]
            )
        ]
        d["s_color"] = [
            puntaje_color_piel(ci, piel, cerca_rostro=(cat in CATEGORIAS_CERCA_ROSTRO))
            for ci, cat in zip(d["color_info"], d["categoria"])
        ]
        d["s_preferencia"] = [
            puntaje_preferencia_color(ci, color_preferido) for ci in d["color_info"]
        ]

        estilo_norm = norm(estilo)
        if estilo_norm in {"casual", "elegante", "formal", "deportivo", "streetwear"}:
            d["s_estilo"] = (d["estilo"] == estilo_norm).astype(float)
        else:
            d["s_estilo"] = 0.5

        if ocasion:
            oc_norm = norm(ocasion)
            s_oc = [
                puntaje_ocasion(est, sub, oc_norm)
                for est, sub in zip(d["estilo"], d["subcategoria"])
            ]
            if estilo_norm in {"casual", "elegante", "formal", "deportivo", "streetwear"}:
                d["s_estilo"] = 0.45 * d["s_estilo"] + 0.55 * pd.Series(s_oc, index=d.index)
            else:
                d["s_estilo"] = s_oc

        # Pesos adaptados por categoria:
        # - top/vestido/abrigo: impactan silueta y tono de piel cerca del rostro.
        # - bottom: prioriza forma corporal, tipo de pantalon/falda y ajuste.
        # - calzado: usa s_forma como calidad/funcion del calzado + estilo + armonia de color.
        pesos = pd.DataFrame(self.pesos, index=d.index, dtype=float)
        es_bottom = d["categoria"] == "bottom"
        es_calzado = d["categoria"].isin(CATEGORIAS_SIN_CUERPO)

        pesos.loc[es_bottom, ["forma", "somato", "color"]] = [0.44, 0.24, 0.17]
        pesos.loc[es_calzado, ["forma", "somato", "color", "estilo"]] = [0.55, 0.0, 0.20, 0.25]

        if not estilo and not ocasion:
            pesos.loc[~es_calzado, "estilo"] = 0.0
            pesos.loc[es_calzado, "estilo"] = 0.0

        pref_activa = bool(color_preferido and norm(color_preferido) != "ninguno")
        if pref_activa:
            pesos["preferencia"] = 0.0
            # El color preferido impulsa sobre todo la prenda protagonista (top/vestido/abrigo),
            # manteniendo pantalones y zapatos neutros disponibles para no saturar el outfit.
            pesos.loc[d["categoria"].isin(CATEGORIAS_CERCA_ROSTRO), "preferencia"] = 0.22
            pesos.loc[es_bottom, "preferencia"] = 0.08

        pesos = pesos.div(pesos.sum(axis=1), axis=0)

        d["score"] = (
            pesos["forma"] * d["s_forma"]
            + pesos["somato"] * d["s_somato"]
            + pesos["color"] * d["s_color"]
            + pesos["estilo"] * d["s_estilo"]
            + pesos.get("preferencia", 0.0) * d["s_preferencia"]
        )
        d.attrs["piel"] = piel
        return d.sort_values("score", ascending=False)

    # ------------------------------------------------------------------
    # recomendar prendas sueltas
    # ------------------------------------------------------------------
    def recomendar_prendas(self, perfil: PerfilUsuario, categoria: Optional[str] = None,
                           n: int = 10, estilo: Optional[str] = None,
                           ocasion: Optional[str] = None,
                           color_preferido: Optional[str] = None) -> list[dict]:
        d = self.puntuar(perfil, estilo, ocasion, color_preferido)
        piel: PerfilPiel = d.attrs["piel"]
        if categoria:
            d = d[d["categoria"] == categoria]
        d = d[d["s_forma"] >= 0.25].drop_duplicates("nombre").head(n)
        return [self._item_dict(r, perfil, piel) for _, r in d.iterrows()]

    # ------------------------------------------------------------------
    # recomendar outfits completos (combinacion inteligente)
    # ------------------------------------------------------------------
    @staticmethod
    def _pool_diverso(
        d: pd.DataFrame,
        categoria: str,
        top_k: int,
        asegurar_neutros: bool = False,
    ) -> list[dict]:
        """Construye un pool variado de prendas de alta puntuacion sin clonar el mismo color/tipo."""
        sub = d[(d["categoria"] == categoria) & (d["s_forma"] >= 0.30)].drop_duplicates("nombre")
        if sub.empty:
            return []

        registros = sub.to_dict("records")
        elegidos: list[dict] = []
        vistos_par: dict[tuple[str, str], int] = {}

        for r in registros:
            clave = (str(r["subcategoria"]), str(r.get("color_hex") or ""))
            if vistos_par.get(clave, 0) >= 3:
                continue
            elegidos.append(r)
            vistos_par[clave] = vistos_par.get(clave, 0) + 1
            if len(elegidos) >= top_k:
                break

        # Garantizar que bottoms, calzados y abrigos tengan algunas bases neutras para combinar
        if asegurar_neutros and len(elegidos) >= 4:
            n_neutros = sum(bool(r.get("color_info") and r["color_info"].es_neutro) for r in elegidos)
            if n_neutros < 4:
                ids_ya = {r["item_id"] for r in elegidos}
                neutros_extra = [
                    r for r in registros
                    if r["item_id"] not in ids_ya
                    and r.get("color_info") is not None
                    and r["color_info"].es_neutro
                ][: 4 - n_neutros]
                if neutros_extra:
                    elegidos = elegidos[: max(2, len(elegidos) - len(neutros_extra))] + neutros_extra

        return elegidos

    def recomendar_outfits(self, perfil: PerfilUsuario, n: int = 5,
                           estilo: Optional[str] = None, top_k: int = 18,
                           incluir_abrigo: bool = True,
                           ocasion: Optional[str] = None,
                           color_preferido: Optional[str] = None) -> list[dict]:
        d = self.puntuar(perfil, estilo, ocasion, color_preferido)
        piel: PerfilPiel = d.attrs["piel"]
        oc_norm = norm(ocasion)
        pref_norm = norm(color_preferido) if color_preferido else ""
        usar_pref = bool(pref_norm and pref_norm != "ninguno")

        tops = self._pool_diverso(d, "top", top_k, asegurar_neutros=False)
        bottoms = self._pool_diverso(d, "bottom", top_k, asegurar_neutros=True)
        vestidos = self._pool_diverso(d, "vestido", max(10, top_k // 2), asegurar_neutros=False)
        calzados = self._pool_diverso(d, "calzado", max(12, top_k - 4), asegurar_neutros=True)
        abrigos = self._pool_diverso(d, "abrigo", max(12, top_k - 4), asegurar_neutros=True)

        combos: list[dict] = []
        if tops and bottoms and calzados:
            combos += [
                {"top": t, "bottom": b, "calzado": c}
                for t, b, c in itertools.product(tops, bottoms, calzados)
            ]
        if vestidos and calzados:
            combos += [
                {"vestido": v, "calzado": c}
                for v, c in itertools.product(vestidos, calzados)
            ]

        evaluados = []
        for c in combos:
            pt, raz = self._evaluar_outfit(
                c, perfil, estilo=estilo, ocasion=ocasion,
                color_preferido=pref_norm if usar_pref else None,
            )
            evaluados.append((pt, c, raz))
        evaluados.sort(key=lambda x: x[0], reverse=True)

        # Control de variedad entre outfits recomendados (incluyendo equilibrio vestido vs top+bottom)
        MAX_USO = {"top": 1, "vestido": 1, "bottom": 1, "calzado": 2, "abrigo": 1}
        max_vestidos = max(1, n // 2) if tops and bottoms else n
        n_vestidos = 0
        elegidos: list[tuple[float, dict, list[str]]] = []
        usos_nombre: dict[str, int] = {}
        usos_id: dict[int, int] = {}

        def libre(cat: str, item: dict) -> bool:
            return usos_id.get(item["item_id"], 0) < MAX_USO[cat]

        def seleccionar(permitir_repetir_bottom: bool) -> None:
            nonlocal n_vestidos
            for puntaje, piezas, razones_out in evaluados:
                if len(elegidos) >= n:
                    break
                if "vestido" in piezas and n_vestidos >= max_vestidos and not permitir_repetir_bottom:
                    continue
                principal = piezas.get("vestido", piezas.get("top"))
                if usos_nombre.get(principal["nombre"], 0) >= 1:
                    continue
                lim_bottom = 2 if permitir_repetir_bottom else MAX_USO["bottom"]
                if any(
                    usos_id.get(p["item_id"], 0) >= (lim_bottom if c == "bottom" else MAX_USO[c])
                    for c, p in piezas.items()
                ):
                    continue

                piezas_final = dict(piezas)
                puntaje_final = puntaje
                razones_final = list(razones_out)

                # Incluir tercera pieza (abrigo/blazer/chaqueta) solo cuando sea coherente
                # con el clima y mejore o mantenga la armonia del conjunto.
                tiene_verano = any(
                    nivel_termico(p["subcategoria"], p["nombre"], c) == "fresco"
                    and norm(p["subcategoria"]) in {"shorts", "flip flop"}
                    for c, p in piezas_final.items()
                )
                # Si el top ya es sueter/sudadera/cardigan/chaqueta, no se agrega otra capa.
                ya_tiene_capa = any(es_capa_exterior(p) for p in piezas_final.values())
                if (incluir_abrigo and abrigos and oc_norm != "playa"
                        and not tiene_verano and not ya_tiene_capa):
                    disponibles = [a for a in abrigos if libre("abrigo", a)]
                    mejor_abrigo = None
                    mejor_pt_abr = -1.0
                    mejor_raz_abr: list[str] = []
                    for a in disponibles:
                        # Verificar que el abrigo sea compatible con cada pieza del outfit
                        min_comp = min(compat_prendas(a, p, ocasion) for p in piezas_final.values())
                        if min_comp < 0.55 and oc_norm != "frio":
                            continue
                        cand = {**piezas_final, "abrigo": a}
                        pt_cand, raz_cand = self._evaluar_outfit(
                            cand, perfil, estilo=estilo, ocasion=ocasion,
                            color_preferido=pref_norm if usar_pref else None,
                        )
                        if pt_cand > mejor_pt_abr:
                            mejor_pt_abr = pt_cand
                            mejor_abrigo = a
                            mejor_raz_abr = raz_cand

                    if mejor_abrigo is not None and (
                        oc_norm == "frio" or mejor_pt_abr >= puntaje_final * 0.94
                    ):
                        piezas_final["abrigo"] = mejor_abrigo
                        puntaje_final = mejor_pt_abr
                        razones_final = mejor_raz_abr

                elegidos.append((puntaje_final, piezas_final, razones_final))
                if "vestido" in piezas_final:
                    n_vestidos += 1
                usos_nombre[principal["nombre"]] = 1
                for c, p in piezas_final.items():
                    usos_id[p["item_id"]] = usos_id.get(p["item_id"], 0) + 1

        seleccionar(permitir_repetir_bottom=False)
        if len(elegidos) < n:
            seleccionar(permitir_repetir_bottom=True)

        elegidos.sort(key=lambda x: x[0], reverse=True)
        return [{
            "puntaje": round(float(pt), 3),
            "piezas": {k: self._item_dict(v, perfil, piel) for k, v in pz.items()},
            "razones": raz,
        } for pt, pz, raz in elegidos]

    # ------------------------------------------------------------------
    # evaluacion integral de un outfit
    # ------------------------------------------------------------------
    @staticmethod
    def _compat_con(prenda: dict, piezas: dict, ocasion: Optional[str] = None) -> float:
        vals = [compat_prendas(prenda, p, ocasion) for p in piezas.values()]
        return sum(vals) / len(vals) if vals else 1.0

    @classmethod
    def _evaluar_outfit(
        cls,
        piezas: dict,
        perfil: Optional[PerfilUsuario] = None,
        estilo: Optional[str] = None,
        ocasion: Optional[str] = None,
        color_preferido: Optional[str] = None,
    ) -> tuple[float, list[str]]:
        razones_outfit: list[str] = []

        # 1. Puntaje individual ponderado (cuerpo + tono de piel)
        pesos = {k: PESO_PIEZA[k] for k in piezas}
        total = sum(pesos.values())
        base = sum(piezas[k]["score"] * w for k, w in pesos.items()) / total

        # 2. Coherencia entre tipos de prenda, estilo y clima
        pares = list(itertools.combinations(piezas.values(), 2))
        comps = [compat_prendas(a, b, ocasion) for a, b in pares]
        coherencia_prendas = sum(comps) / len(comps) if comps else 1.0
        min_comp = min(comps, default=1.0)

        # Si dos prendas chocan gravemente (ej. abrigo invernal + shorts o blazer + chanclas),
        # penalizamos todo el conjunto para que nunca salga elegido.
        penal = 1.0
        if min_comp < 0.45:
            penal *= 0.58
        elif min_comp < 0.65:
            penal *= 0.84
        elif coherencia_prendas >= 0.88:
            razones_outfit.append("prendas coherentes en estilo, formalidad y clima")

        # 3. Armonia cromatica global del conjunto
        mapa_colores = {k: p.get("color_info") for k, p in piezas.items()}
        armonia_col, raz_col = armonia_outfit_colores(mapa_colores)
        razones_outfit.extend(raz_col)

        # 4. Equilibrio de silueta y color segun el tipo de cuerpo del usuario (0..1)
        s_silueta = 0.85
        if perfil is not None:
            s_silueta, raz_cuerpo = equilibrio_cuerpo_outfit(
                piezas, perfil.genero, perfil.forma, perfil.somatotipo, estilo
            )
            razones_outfit.extend(raz_cuerpo)

        # 5. Control de estampados (maximo 1 prenda estampada protagonista)
        n_estamp = sum(bool(p.get("estampado")) for p in piezas.values())
        penal *= {0: 1.0, 1: 1.0, 2: 0.78}.get(n_estamp, 0.55)

        puntaje = (
            0.40 * base
            + 0.22 * coherencia_prendas
            + 0.22 * armonia_col
            + 0.16 * s_silueta
        ) * penal

        # 6. Bonus suave si al menos una pieza principal refleja el color preferido del usuario
        if color_preferido:
            principales = [piezas[k] for k in ("top", "vestido", "abrigo", "bottom") if k in piezas]
            if any(p.get("s_preferencia", 0.5) >= 0.85 for p in principales):
                puntaje = min(0.995, puntaje + 0.025)
            else:
                puntaje *= 0.88

        return min(1.0, puntaje), razones_outfit

    @classmethod
    def _puntaje_outfit(cls, piezas: dict, perfil: Optional[PerfilUsuario] = None) -> float:
        pt, _ = cls._evaluar_outfit(piezas, perfil=perfil)
        return pt

    def _ruta_imagen(self, p: str) -> str:
        return str(self.imagenes_dir / Path(str(p).replace("\\", "/")).name)

    def _item_dict(self, r, perfil: PerfilUsuario, piel: Optional[PerfilPiel] = None) -> dict:
        piel_obj = piel or perfil_piel(perfil.piel_hex)
        razones: list[str] = []
        sub_es = _SUBCAT_ES.get(norm(r["subcategoria"]), str(r["subcategoria"]).lower())
        cat = r["categoria"]

        if cat not in CATEGORIAS_SIN_CUERPO:
            if r["s_forma"] >= 0.82:
                corte_txt = f" de corte {r['corte']}" if r["corte"] != "sin_corte" else ""
                razones.append(f"{sub_es}{corte_txt} favorece tu silueta ({perfil.forma})")
            if r["s_somato"] >= 0.82:
                razones.append(f"ajuste {r['ajuste']} equilibrado para tu somatotipo ({perfil.somatotipo})")

        ci = r.get("color_info")
        if cat in CATEGORIAS_CERCA_ROSTRO and r["s_color"] >= 0.80:
            razones.append(
                f"color {r['color_es']} armoniza con tu piel {piel_obj.profundidad} ({piel_obj.subtono})"
            )
        elif cat not in CATEGORIAS_CERCA_ROSTRO and ci is not None and ci.es_neutro:
            razones.append(f"tono neutro ({r['color_es']}) facil de combinar en el conjunto")
        elif r["s_color"] >= 0.82:
            razones.append(f"color {r['color_es']} complementa tu paleta personal")

        return {
            "item_id": int(r["item_id"]),
            "nombre": r["nombre"],
            "categoria": cat,
            "subcategoria": r["subcategoria"],
            "estilo": r["estilo"],
            "corte": r["corte"],
            "ajuste": r["ajuste"],
            "color": r["color_es"],
            "color_hex": r["color_hex"],
            "imagen": self._ruta_imagen(r["imagen_path"]),
            "score": round(float(r["score"]), 3),
            "scores": {k: round(float(r[k]), 3)
                       for k in ("s_forma", "s_somato", "s_color", "s_estilo")},
            "razones": razones,
        }
