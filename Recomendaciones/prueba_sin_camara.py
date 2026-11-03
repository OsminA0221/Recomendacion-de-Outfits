"""PASO 3: prueba el recomendador SIN camara, con perfiles representativos.

Uso (desde la raiz del proyecto):   python -m Recomendaciones.prueba_sin_camara
Sirve para comprobar que las recomendaciones tienen sentido estilistico y
colorimetrico antes de conectar la camara. Guarda un collage por perfil en salida/.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from .recomendador import Recomendador, perfil_desde_dict
from .reglas import es_capa_exterior, nivel_termico, norm
from .visual import crear_collage

PERFILES = {
    "Mujer reloj de arena, mesomorfo, piel clara fria": {
        "genero": "Mujer", "forma_corporal": {"label": "Reloj de arena"},
        "somatotipo": {"label": "Mesomorfo", "confidence": 0.8},
        "tono_piel": {"label": "CJ", "skin_tone": "#f3dad6"}},
    "Mujer pera, ectomorfo, piel media calida": {
        "genero": "Mujer", "forma_corporal": {"label": "Triangulo / Pera"},
        "somatotipo": {"label": "Ectomorfo", "confidence": 0.7},
        "tono_piel": {"label": "CG", "skin_tone": "#bea07e"}},
    "Mujer manzana, endomorfo, piel oscura": {
        "genero": "Mujer", "forma_corporal": {"label": "Ovalado / Manzana"},
        "somatotipo": {"label": "Endomorfo", "confidence": 0.6},
        "tono_piel": {"label": "CD", "skin_tone": "#6f503c"}},
    "Hombre trapecio, mesomorfo, piel media calida": {
        "genero": "Hombre", "forma_corporal": {"label": "Trapecio"},
        "somatotipo": {"label": "Mesomorfo", "confidence": 0.9},
        "tono_piel": {"label": "CF", "skin_tone": "#9d7a54"}},
    "Hombre rectangulo, ectomorfo, piel clara calida": {
        "genero": "Hombre", "forma_corporal": {"label": "Rectangulo"},
        "somatotipo": {"label": "Ectomorfo", "confidence": 0.75},
        "tono_piel": {"label": "CH", "skin_tone": "#e5c8a6"}},
    "Hombre ovalo, endomorfo, piel oscura": {
        "genero": "Hombre", "forma_corporal": {"label": "Ovalo"},
        "somatotipo": {"label": "Endomorfo", "confidence": 0.55},
        "tono_piel": {"label": "CC", "skin_tone": "#513b2e"}},
}


def main() -> None:
    rec = Recomendador()
    salida = Path(__file__).resolve().parent / "salida"
    salida.mkdir(exist_ok=True)

    for i, (nombre, d) in enumerate(PERFILES.items(), 1):
        perfil = perfil_desde_dict(d)
        outfits = rec.recomendar_outfits(perfil, n=3)
        print("=" * 78)
        print(nombre)
        for k, o in enumerate(outfits, 1):
            print(f"  Outfit {k}  puntaje {o['puntaje']}")
            for r_out in o.get("razones", []):
                print(f"     * {r_out}")
            for cat, p in o["piezas"].items():
                print(f"     {cat:8s} [{p['subcategoria'][:14]:14s}] {p['nombre'][:24]:24s} "
                      f"{p['color']:14s} {p['corte']:9s}/{p['ajuste']:8s} score {p['score']}")
                for r in p["razones"][:2]:
                    print(f"              - {r}")
            # Comprobaciones de sanidad estilistica y funcional
            assert all(x["categoria"] in ("top", "bottom", "vestido", "calzado", "abrigo")
                       for x in o["piezas"].values())
            assert all(norm(x["subcategoria"]) not in ("slippers", "costumes", "flip flop")
                       for x in o["piezas"].values())
            niveles = {
                nivel_termico(x["subcategoria"], x["nombre"], x["categoria"])
                for x in o["piezas"].values()
            }
            subs = {norm(x["subcategoria"]) for x in o["piezas"].values()}
            if not ("boots" in subs and (subs & {"dress", "skirt"})):
                assert niveles != {"fresco", "abrigo_frio"}, f"Choque termico en {o['piezas']}"
            if perfil.genero == "Hombre":
                assert "vestido" not in o["piezas"]
            # Maximo UNA capa que abriga (sueter/sudadera/cardigan/chaqueta/abrigo/blazer)
            capas = [x["nombre"] for x in o["piezas"].values() if es_capa_exterior(x)]
            assert len(capas) <= 1, f"Mas de una capa en el outfit: {capas}"
        cv2.imwrite(str(salida / f"perfil_{i}.jpg"), crear_collage(outfits))
    print("\nCollages guardados en", salida)


if __name__ == "__main__":
    main()
