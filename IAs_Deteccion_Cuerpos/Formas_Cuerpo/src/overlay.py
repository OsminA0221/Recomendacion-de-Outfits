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