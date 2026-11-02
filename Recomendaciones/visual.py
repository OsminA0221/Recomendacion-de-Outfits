"""Dibuja los outfits recomendados como un collage (para cv2.imshow)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TAM = 200          # lado de cada tarjeta de prenda
ORDEN = ("vestido", "top", "bottom", "abrigo", "calzado")


def _hex_a_bgr(h):
    if not isinstance(h, str):
        return (170, 170, 170)
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def _tarjeta(pieza: dict) -> np.ndarray:
    ruta = pieza.get("imagen")
    img = cv2.imread(ruta) if ruta and Path(ruta).exists() else None
    if img is None:      # sin imagen: cuadro del color de la prenda
        img = np.full((TAM, TAM, 3), _hex_a_bgr(pieza.get("color_hex")), np.uint8)
    h, w = img.shape[:2]
    esc = TAM / max(h, w)
    img = cv2.resize(img, (int(w * esc), int(h * esc)))
    lienzo = np.full((TAM + 34, TAM, 3), 245, np.uint8)
    y0, x0 = (TAM - img.shape[0]) // 2, (TAM - img.shape[1]) // 2
    lienzo[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    texto = f"{pieza['categoria']}: {pieza['nombre']}"[:30]
    cv2.putText(lienzo, texto, (4, TAM + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(lienzo, f"{pieza['color']} | {pieza['score']:.2f}", (4, TAM + 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    return lienzo


def crear_collage(outfits: list[dict]) -> np.ndarray:
    """Una fila por outfit: [numero + puntaje] [prenda] [prenda] ..."""
    if not outfits:
        vacio = np.full((120, 520, 3), 245, np.uint8)
        cv2.putText(vacio, "Sin outfits para este perfil", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        return vacio
    filas = []
    for i, o in enumerate(outfits, 1):
        cab = np.full((TAM + 34, 90, 3), 230, np.uint8)
        cv2.putText(cab, f"#{i}", (18, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(cab, f"{o['puntaje']:.2f}", (18, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 1, cv2.LINE_AA)
        cartas = [cab] + [_tarjeta(o["piezas"][c]) for c in ORDEN if c in o["piezas"]]
        filas.append(np.hstack(cartas))
    ancho = max(f.shape[1] for f in filas)
    filas = [np.hstack([f, np.full((f.shape[0], ancho - f.shape[1], 3), 245, np.uint8)]) for f in filas]
    return np.vstack(filas)