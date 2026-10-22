from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    gender: str = "Mujer"             # Género inicial: "Mujer" o "Hombre"

    ttl_seconds: float = 300.0        # el perfil guardado vive 5 min
    refresh_seconds: float = 60.0     # se intenta re-guardar cada 1 min
    stable_seconds: float = 2.0       # margen: detectar bien X s seguidos (2 o 5)
    error_grace_seconds: float = 0.5  # tolerancia a un ciclo malo suelto

    analysis_period: float = 0.25     # pausa mínima entre ciclos

    min_somato_conf: float = 0.25
    min_skin_accuracy: float = 70.0

    body_smoothing_alpha: float = 0.5
