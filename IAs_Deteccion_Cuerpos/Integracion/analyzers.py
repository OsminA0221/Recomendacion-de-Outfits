from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

# stone consulta PyPI al importarse; lo desactivamos para no demorar el arranque
os.environ.setdefault("STONE_UPGRADE_FLAG", "1")

_ROOT = Path(__file__).resolve().parents[1]          # IAs_Deteccion_Cuerpos/
FORMAS_DIR = _ROOT / "Formas_Cuerpo"
SOMATO_WEIGHTS = _ROOT / "SomatoSense" / "best.pt"


@dataclass
class AnalyzerResult:
    ok: bool
    data: dict = field(default_factory=dict)
    reason: str = ""
    latency: float = 0.0


def _ok(data: dict) -> AnalyzerResult:
    return AnalyzerResult(True, data=data)


def _fail(reason: str) -> AnalyzerResult:
    return AnalyzerResult(False, reason=reason)



class BaseAnalyzer:
    name = "base"

    def analyze(self, frame) -> AnalyzerResult:
        raise NotImplementedError

    def run(self, frame) -> AnalyzerResult:
        """analyze() + medición de tiempo + captura de excepciones."""
        t0 = time.perf_counter()
        try:
            result = self.analyze(frame)
        except Exception as exc:  # una IA caída no debe tumbar a las demás
            result = _fail(f"error interno: {type(exc).__name__}: {exc}")
        result.latency = time.perf_counter() - t0
        return result

    def close(self) -> None:
        pass


class BodyShapeAnalyzer(BaseAnalyzer):
    name = "forma"

    def __init__(self, smoothing_alpha: float):
        if str(FORMAS_DIR) not in sys.path:
            sys.path.insert(0, str(FORMAS_DIR))
        from src.pose_detector import PoseDetector
        from src.body_measurements import (
            BodyMeasurementSmoother,
            compute_proportions,
        )
        from src.body_shape_classifier import classify_body_shape

        self._compute = compute_proportions
        self._classify = classify_body_shape
        self.detector = PoseDetector()
        self.smoother = BodyMeasurementSmoother.create(alpha=smoothing_alpha)

    def analyze(self, frame) -> AnalyzerResult:
        landmarks = self.detector.process(frame)

        if landmarks is None:
            self.smoother.reset()
            return _fail("no se detecta una persona")

        if not landmarks.visibility_ok:
            self.smoother.reset()
            return _fail("cuerpo incompleto (muestrate de cuerpo completo)")

        props = self._compute(
            landmarks, detector=self.detector, smoother=self.smoother
        )

        # Si la medición cayó al respaldo por landmarks, no es confiable
        if not props.measured_from_silhouette:
            self.smoother.reset()
            return _fail("no se pudo medir la silueta")

        shape = self._classify(props)
        return _ok({
            "label": shape.label,
            "shoulder_to_hip": shape.shoulder_to_hip_ratio,
            "waist_to_hip": shape.waist_to_hip_ratio,
            # copia en píxeles para dibujar desde el hilo principal
            "points": list(self.detector._last_landmarks_px or []),
        })

class SomatotypeAnalyzer(BaseAnalyzer):
    name = "somatotipo"

    def __init__(self, weights: Path, min_conf: float):
        from ultralytics import YOLO

        if not Path(weights).exists():
            raise FileNotFoundError(f"No se encontró {weights}")
        self.model = YOLO(str(weights))
        self.min_conf = min_conf

    def analyze(self, frame) -> AnalyzerResult:
        result = self.model.predict(
            frame, conf=self.min_conf, verbose=False
        )[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return _fail(f"sin somatotipo con conf >= {self.min_conf:.2f}")

        i = int(boxes.conf.argmax())
        x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i].tolist())
        return _ok({
            "label": result.names[int(boxes.cls[i])],
            "confidence": round(float(boxes.conf[i]), 3),
            "bbox": (x1, y1, x2, y2),
        })
    

class SkinToneAnalyzer(BaseAnalyzer):
    name = "piel"

    def __init__(self, min_accuracy: float):
        from stone.image import (
            build_full_palette,
            default_tone_labels,
            process_image,
        )

        self._process_image = process_image
        self._palette = build_full_palette()["perla"]
        self._labels = default_tone_labels(self._palette, "C")
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.min_accuracy = min_accuracy

    def analyze(self, frame) -> AnalyzerResult:
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return _fail("no se detecta rostro")

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Recorte con margen para que stone vuelva a encontrar la cara
        m = int(0.4 * max(w, h))
        H, W = frame.shape[:2]
        x1, y1 = max(0, x - m), max(0, y - m)
        x2, y2 = min(W, x + w + m), min(H, y + h + m)
        crop = frame[y1:y2, x1:x2]

        records, _ = self._process_image(
            crop, False, False,
            self._palette, self._labels,
            new_width=250,
            n_dominant_colors=2,
            minSize=(60, 60),
            threshold=0.15,
        )
        rec = records[0]

        # "NA" = stone no halló cara y clasificó toda la imagen: se rechaza
        if rec.get("face_id") == "NA":
            return _fail("no se pudo aislar la piel del rostro")

        accuracy = float(rec["accuracy"])
        if accuracy < self.min_accuracy:
            return _fail(
                f"precision de piel baja ({accuracy:.0f} < "
                f"{self.min_accuracy:.0f})"
            )

        return _ok({
            "skin_tone": rec["skin_tone"],
            "label": rec["tone_label"],
            "accuracy": accuracy,
            "bbox": (int(x), int(y), int(x + w), int(y + h)),
        })