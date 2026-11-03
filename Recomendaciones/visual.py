"""Dibuja los outfits recomendados como un collage explicable (para cv2.imshow)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TAM = 200          # lado de cada tarjeta de prenda
ORDEN = ("vestido", "top", "bottom", "abrigo", "calzado")


def _hex_a_bgr(h) -> tuple[int, int, int]:
    if not isinstance(h, str) or len(h.lstrip("#")) < 6:
        return (170, 170, 170)
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def _tarjeta(pieza: dict) -> np.ndarray:
    ruta = pieza.get("imagen")
    img = cv2.imread(ruta) if ruta and Path(ruta).exists() else None
    bgr_prenda = _hex_a_bgr(pieza.get("color_hex"))
    if img is None:      # sin imagen: cuadro del color de la prenda
        img = np.full((TAM, TAM, 3), bgr_prenda, np.uint8)
    h, w = img.shape[:2]
    esc = TAM / max(h, w)
    img = cv2.resize(img, (int(w * esc), int(h * esc)))
    lienzo = np.full((TAM + 46, TAM, 3), 245, np.uint8)
    y0, x0 = (TAM - img.shape[0]) // 2, (TAM - img.shape[1]) // 2
    lienzo[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img

    subcat = pieza.get("subcategoria") or pieza["categoria"]
    texto1 = f"{subcat}: {pieza['nombre']}"[:28]
    texto2 = f"{pieza['color']} ({pieza['corte']})"[:28]
    cv2.putText(lienzo, texto1, (4, TAM + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.circle(lienzo, (10, TAM + 28), 5, bgr_prenda, -1)
    cv2.circle(lienzo, (10, TAM + 28), 5, (80, 80, 80), 1)
    cv2.putText(lienzo, texto2, (20, TAM + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(lienzo, f"score {pieza['score']:.2f}", (4, TAM + 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1, cv2.LINE_AA)
    return lienzo


def crear_collage(outfits: list[dict]) -> np.ndarray:
    """Una fila por outfit: [numero + puntaje] [prendas...] + barra de razon sin recortar."""
    if not outfits:
        vacio = np.full((120, 520, 3), 245, np.uint8)
        cv2.putText(vacio, "Sin outfits para este perfil", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        return vacio

    filas_cartas = []
    for i, o in enumerate(outfits, 1):
        cab = np.full((TAM + 46, 94, 3), 230, np.uint8)
        cv2.putText(cab, f"#{i}", (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(cab, f"{o['puntaje']:.2f}", (18, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (80, 80, 80), 1, cv2.LINE_AA)
        cartas = [cab] + [_tarjeta(o["piezas"][c]) for c in ORDEN if c in o["piezas"]]
        filas_cartas.append((np.hstack(cartas), o.get("razones") or []))

    ancho = max(f.shape[1] for f, _ in filas_cartas)
    max_chars = max(45, (ancho - 20) // 7)

    filas_finales = []
    for fila_img, razones in filas_cartas:
        if fila_img.shape[1] < ancho:
            relleno = np.full((fila_img.shape[0], ancho - fila_img.shape[1], 3), 245, np.uint8)
            fila_img = np.hstack([fila_img, relleno])
        barra = np.full((24, ancho, 3), 236, np.uint8)
        if razones:
            # Priorizar las razones especificas de silueta y color
            ordenadas = list(reversed(razones))
            partes = []
            for r in ordenadas:
                cand = " | ".join(partes + [r])
                if len(cand) <= max_chars:
                    partes.append(r)
            resumen = " | ".join(partes) if partes else ordenadas[0][:max_chars]
            cv2.putText(barra, resumen, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (45, 75, 45), 1, cv2.LINE_AA)
        filas_finales.append(np.vstack([fila_img, barra]))

    return np.vstack(filas_finales)
