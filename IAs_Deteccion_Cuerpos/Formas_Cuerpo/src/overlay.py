"""Dibuja los resultados calculados sobre el frame."""

import cv2
import numpy as np

from src.body_measurements import BodyProportions
from src.body_shape_classifier import BodyShapeResult


_FONT = cv2.FONT_HERSHEY_SIMPLEX

_TEXT_COLOR = (0, 255, 0)

_WARN_COLOR = (0, 165, 255)

_INFO_COLOR = (200, 200, 200)

_SHOULDER_COLOR = (255, 0, 0)

_WAIST_COLOR = (0, 255, 255)

_HIP_COLOR = (255, 0, 255)


def _draw_measurement_line(
    frame: np.ndarray,
    y_normalized: float,
    label: str,
    color: tuple
) -> None:

    h, w = frame.shape[:2]

    y = int(
        np.clip(
            y_normalized,
            0.0,
            1.0
        ) * (h - 1)
    )

    cv2.line(
        frame,
        (0, y),
        (w - 1, y),
        color,
        2
    )

    cv2.putText(
        frame,
        label,
        (15, max(20, y - 5)),
        _FONT,
        0.55,
        color,
        2
    )


def _edge_points(frame: np.ndarray, edges):
    """((x,y),(x,y)) normalizados -> dos puntos en píxeles (o None)."""
    if edges is None:
        return None
    h, w = frame.shape[:2]
    (xl, yl), (xr, yr) = edges
    return (
        (int(xl * (w - 1)), int(yl * (h - 1))),
        (int(xr * (w - 1)), int(yr * (h - 1))),
    )


def _draw_edge_row(
    frame: np.ndarray,
    points,
    color: tuple,
    label: str
) -> None:
    """Segmento entre los dos extremos de la silueta + un punto en cada uno."""
    if points is None:
        return

    left, right = points
    cv2.line(frame, left, right, color, 3)

    for pt in (left, right):
        cv2.circle(frame, pt, 9, (255, 255, 255), -1)
        cv2.circle(frame, pt, 6, color, -1)

    h, w = frame.shape[:2]
    text = f"{label} {right[0] - left[0]}px"
    text_x = int(np.clip(right[0] + 14, 0, w - 170))
    text_y = max(15, right[1] - 12)   # arriba de la línea para que se lea
    cv2.putText(frame, text, (text_x, text_y), _FONT, 0.5, (0, 0, 0), 4)
    cv2.putText(frame, text, (text_x, text_y), _FONT, 0.5, color, 2)


def _draw_edge_skeleton(
    frame: np.ndarray,
    proportions: BodyProportions
) -> None:
    """Dibuja hombros, cintura y cadera EN LOS EXTREMOS de la silueta,
    que es exactamente entre donde se miden los anchos."""

    shoulders = _edge_points(frame, proportions.shoulder_edges)
    waist = _edge_points(frame, proportions.waist_edges)
    hips = _edge_points(frame, proportions.hip_edges)

    for side in (0, 1):
        chain = [r[side] for r in (shoulders, waist, hips) if r is not None]
        for a, b in zip(chain, chain[1:]):
            cv2.line(frame, a, b, (0, 200, 0), 2)

    _draw_edge_row(frame, shoulders, _SHOULDER_COLOR, "hombros")
    _draw_edge_row(frame, waist, _WAIST_COLOR, "cintura")
    _draw_edge_row(frame, hips, _HIP_COLOR, "cadera")


def draw_results(
    frame: np.ndarray,
    proportions: BodyProportions,
    shape_result: BodyShapeResult
) -> np.ndarray:

    _draw_measurement_line(
        frame,
        proportions.shoulder_y,
        "HOMBROS",
        _SHOULDER_COLOR
    )

    _draw_measurement_line(
        frame,
        proportions.waist_y,
        "REFERENCIA CINTURA",
        _WAIST_COLOR
    )

    _draw_measurement_line(
        frame,
        proportions.hip_y,
        "CADERA",
        _HIP_COLOR
    )

    _draw_edge_skeleton(frame, proportions)

    metodo = (
        "silueta 2D"
        if proportions.measured_from_silhouette
        else "landmarks"
    )

    lines = [

        f"Forma corporal: "
        f"{shape_result.label}",

        f"Ratio hombro/cadera: "
        f"{shape_result.shoulder_to_hip_ratio}",

        f"Ratio cintura/cadera: "
        f"{shape_result.waist_to_hip_ratio}",

    ]

    y = 30

    for line in lines:

        color = (
            _INFO_COLOR
            if line.startswith("Metodo")
            else _TEXT_COLOR
        )

        cv2.putText(
            frame,
            line,
            (15, y),
            _FONT,
            0.65,
            color,
            2
        )

        y += 28

    return frame


def draw_no_detection_warning(
    frame: np.ndarray
) -> np.ndarray:

    cv2.putText(
        frame,
        (
            "Cuerpo no detectado. "
            "Ubicate de cuerpo completo "
            "frente a la camara."
        ),
        (15, 30),
        _FONT,
        0.6,
        _WARN_COLOR,
        2,
    )

    return frame