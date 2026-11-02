"""PASO 3: prueba el recomendador SIN camara, con perfiles inventados.

Uso (desde la raiz del proyecto):   python -m Recomendaciones.probar_sin_camara
Sirve para comprobar que las recomendaciones tienen sentido antes de
conectar la camara. Tambien guarda un collage por perfil en salida/.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from .recomendador import Recomendador, perfil_desde_dict
from .visual import crear_collage

PERFILES = {
    "Mujer reloj de arena, mesomorfo, piel clara": {
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
    "Hombre trapecio, mesomorfo, piel media": {
        "genero": "Hombre", "forma_corporal": {"label": "Trapecio"},
        "somatotipo": {"label": "Mesomorfo", "confidence": 0.9},
        "tono_piel": {"label": "CF", "skin_tone": "#9d7a54"}},
    "Hombre rectangulo, ectomorfo, piel clara": {
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
        print("=" * 72)
        print(nombre)
        for k, o in enumerate(outfits, 1):
            print(f"  Outfit {k}  puntaje {o['puntaje']}")
            for cat, p in o["piezas"].items():
                print(f"     {cat:8s} {p['nombre'][:28]:28s} {p['color']:11s} "
                      f"{p['corte']:10s}/{p['ajuste']:8s} score {p['score']}")
                for r in p["razones"][:2]:
                    print(f"              - {r}")
            # comprobaciones de sanidad
            assert all(x["categoria"] in ("top", "bottom", "vestido", "calzado", "abrigo")
                       for x in o["piezas"].values())
            if perfil.genero == "Hombre":
                assert "vestido" not in o["piezas"]
        cv2.imwrite(str(salida / f"perfil_{i}.jpg"), crear_collage(outfits))
    print("\nCollages guardados en", salida)


if __name__ == "__main__":
    main()