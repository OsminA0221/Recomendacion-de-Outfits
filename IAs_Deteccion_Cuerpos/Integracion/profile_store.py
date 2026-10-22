from __future__ import annotations

import statistics
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from analyzers import AnalyzerResult

ANALYZER_NAMES = ("forma", "somatotipo", "piel")


@dataclass
class Profile:
    forma: dict
    somatotipo: dict
    piel: dict
    saved_at: datetime
    genero: str = "Mujer"

    def to_dict(self) -> dict:
        return {
            "genero": self.genero,
            "forma_corporal": self.forma,
            "somatotipo": self.somatotipo,
            "tono_piel": self.piel,
            "guardado_en": self.saved_at.isoformat(timespec="seconds"),
        }


@dataclass
class Status:
    state: str            # ESPERANDO | ESTABILIZANDO | GUARDADO | REFRESCO_PENDIENTE
    progress: float       # 0..1 del margen de estabilidad
    failing: dict         # nombre -> motivo (solo los que fallan ahora)
    profile: Optional[Profile]
    expires_in: Optional[float]
    refresh_in: Optional[float]
    last_results: dict    # nombre -> AnalyzerResult (último ciclo)
    gender: str = "Mujer"


def _mode(values: list):
    """Valor más frecuente; en empate gana el más reciente."""
    counts = Counter(values)
    best = max(counts.values())
    for v in reversed(values):
        if counts[v] == best:
            return v


def _consensus(samples: list[dict]) -> tuple[dict, dict, dict]:
    """Une varias muestras en un resultado estable (moda / mediana)."""
    labels = [s["forma"]["label"] for s in samples]
    label = _mode(labels)
    same = [s["forma"] for s in samples if s["forma"]["label"] == label]
    forma = {
        "label": label,
        "gender": same[-1].get("gender", "Mujer"),
        "shoulder_to_hip": round(
            statistics.median(f["shoulder_to_hip"] for f in same), 3),
        "waist_to_hip": round(
            statistics.median(f["waist_to_hip"] for f in same), 3),
    }

    label = _mode([s["somatotipo"]["label"] for s in samples])
    same = [s["somatotipo"] for s in samples if s["somatotipo"]["label"] == label]
    somato = {
        "label": label,
        "confidence": round(statistics.mean(x["confidence"] for x in same), 3),
    }

    label = _mode([s["piel"]["label"] for s in samples])
    same = [s["piel"] for s in samples if s["piel"]["label"] == label]
    piel = {
        "label": label,
        "skin_tone": same[-1]["skin_tone"],
        "accuracy": round(statistics.mean(x["accuracy"] for x in same), 2),
    }
    return forma, somato, piel


class ProfileManager:
    def __init__(
        self,
        ttl_seconds: float,
        refresh_seconds: float,
        stable_seconds: float,
        error_grace_seconds: float,
        initial_gender: str = "Mujer",
        on_save: Optional[Callable[[Profile], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.ttl = ttl_seconds
        self.refresh = refresh_seconds
        self.stable = stable_seconds
        self.grace = error_grace_seconds
        self.gender = initial_gender
        self._on_save = on_save
        self._clock = clock

        self._lock = threading.Lock()
        self._profile: Optional[Profile] = None
        self._saved_at: Optional[float] = None

        self._samples: list[tuple[float, dict]] = []
        self._streak_start: Optional[float] = None
        self._last_good: Optional[float] = None
        self._last_results: dict = {}

    def set_gender(self, gender: str) -> None:
        with self._lock:
            if self.gender != gender:
                self.gender = gender
                self._profile = None
                self._saved_at = None
                self._reset_streak()

    def update(self, results: dict[str, AnalyzerResult]) -> None:
        """Llamar una vez por ciclo con el resultado de las 3 IAs."""
        saved: Optional[Profile] = None
        with self._lock:
            now = self._clock()
            self._last_results = results
            self._expire(now)

            if all(r.ok for r in results.values()):
                if self._streak_start is None:
                    self._streak_start = now
                self._last_good = now
                self._samples.append(
                    (now, {k: r.data for k, r in results.items()}))
                self._samples = [
                    (t, d) for t, d in self._samples if now - t <= self.stable
                ]
            elif (
                self._streak_start is not None
                and now - self._last_good > self.grace
            ):
                self._reset_streak()

            saved = self._maybe_commit(now)

        if saved is not None and self._on_save:   # fuera del lock
            self._on_save(saved)

    def get_profile(self) -> Optional[Profile]:
        """Perfil vigente (o None si no hay o ya expiró)."""
        with self._lock:
            self._expire(self._clock())
            return self._profile

    def clear(self) -> None:
        with self._lock:
            self._profile = None
            self._saved_at = None
            self._reset_streak()

    def status(self) -> Status:
        with self._lock:
            now = self._clock()
            self._expire(now)

            failing = {
                k: r.reason
                for k, r in self._last_results.items() if not r.ok
            }
            refresh_in = (
                max(0.0, self.refresh - (now - self._saved_at))
                if self._saved_at is not None else None
            )
            expires_in = (
                max(0.0, self.ttl - (now - self._saved_at))
                if self._saved_at is not None else None
            )

            progress = 0.0
            if self._streak_start is not None and self.stable > 0:
                progress = min(
                    1.0, (self._last_good - self._streak_start) / self.stable)

            if self._profile is None:
                state = "ESTABILIZANDO" if self._streak_start else "ESPERANDO"
            elif refresh_in and refresh_in > 0:
                state = "GUARDADO"
            else:
                state = ("ESTABILIZANDO" if self._streak_start
                         else "REFRESCO_PENDIENTE")

            return Status(state, progress, failing, self._profile,
                          expires_in, refresh_in, dict(self._last_results),
                          gender=self.gender)

    def _reset_streak(self) -> None:
        self._streak_start = None
        self._last_good = None
        self._samples = []

    def _expire(self, now: float) -> None:
        if self._saved_at is not None and now - self._saved_at >= self.ttl:
            self._profile = None
            self._saved_at = None

    def _maybe_commit(self, now: float) -> Optional[Profile]:
        if self._streak_start is None:
            return None
        if self._last_good - self._streak_start < self.stable:
            return None            # aún no se cumple el margen
        if self._saved_at is not None and now - self._saved_at < self.refresh:
            return None            # aún no toca refrescar

        forma, somato, piel = _consensus([d for _, d in self._samples])
        self._profile = Profile(forma, somato, piel, datetime.now(), genero=self.gender)
        self._saved_at = now
        self._reset_streak()       # el siguiente guardado exige margen nuevo
        return self._profile
