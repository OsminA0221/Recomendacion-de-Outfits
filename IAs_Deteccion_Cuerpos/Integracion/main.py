from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

import cv2

from analyzers import (
    SOMATO_WEIGHTS,
    BodyShapeAnalyzer,
    SkinToneAnalyzer,
    SomatotypeAnalyzer,
)
from config import Config
from pipeline import AnalysisPipeline
from profile_store import ANALYZER_NAMES, Profile, ProfileManager
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RUTA_SRC_STONE = BASE_DIR / "Tonalidades_Piel" / "src"

if str(RUTA_SRC_STONE) not in sys.path:
    sys.path.insert(0, str(RUTA_SRC_STONE))

WINDOW = "Outfit AI - 3 IAs en paralelo"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, YELLOW, WHITE = (0, 220, 0), (0, 0, 255), (0, 215, 255), (255, 255, 255)

LABELS = {"forma": "Forma", "somatotipo": "Somatotipo", "piel": "Piel"}


def parse_args(cfg: Config) -> Config:
    p = argparse.ArgumentParser(description="3 IAs en paralelo con camara en vivo")
    p.add_argument("--camera", type=int, default=cfg.camera_index)
    p.add_argument("--stable", type=float, default=cfg.stable_seconds,
                   help="segundos de deteccion estable antes de guardar (2 o 5)")
    p.add_argument("--refresh", type=float, default=cfg.refresh_seconds,
                   help="cada cuantos segundos se vuelve a guardar")
    p.add_argument("--ttl", type=float, default=cfg.ttl_seconds,
                   help="segundos que vive el perfil guardado")
    a = p.parse_args()
    return replace(cfg, camera_index=a.camera, stable_seconds=a.stable,
                   refresh_seconds=a.refresh, ttl_seconds=a.ttl)


def fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def put(frame, text, y, color=WHITE, scale=0.55):
    cv2.putText(frame, text, (12, y), FONT, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (12, y), FONT, scale, color, 1, cv2.LINE_AA)


def short(data: dict, name: str) -> str:
    if name == "forma":
        return f"{data['label']} (h/c {data['shoulder_to_hip']}, c/c {data['waist_to_hip']})"
    if name == "somatotipo":
        return f"{data['label']} {data['confidence'] * 100:.0f}%"
    return f"{data['label']} {data['skin_tone']} prec {data['accuracy']:.0f}"


def draw_hud(frame, st, cycle_ms):
    y = 24
    for name in ANALYZER_NAMES:
        r = st.last_results.get(name)
        if r is None:
            put(frame, f"[..] {LABELS[name]}: iniciando", y, YELLOW)
        elif r.ok:
            put(frame, f"[OK] {LABELS[name]}: {short(r.data, name)}", y, GREEN)
        else:
            put(frame, f"[X ] {LABELS[name]}: {r.reason}", y, RED)
        y += 24

    for name, color in (("somatotipo", (255, 128, 0)), ("piel", (255, 0, 255))):
        r = st.last_results.get(name)
        if r and r.ok:
            x1, y1, x2, y2 = r.data["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        r = st.last_results.get("forma")
    if r and r.ok:
        rows = (("hombros", (255, 0, 0)), ("cintura", (0, 255, 255)),
                ("cadera", (255, 0, 255)))
        for key, color in rows:
            e = r.data["edges"].get(key)
            if e:
                cv2.line(frame, e[0], e[1], color, 3)
                for pt in e:
                    cv2.circle(frame, pt, 7, (255, 255, 255), -1)
                    cv2.circle(frame, pt, 5, color, -1)

    h = frame.shape[0]
    if st.state == "ESPERANDO":
        put(frame, "Esperando que las 3 IAs detecten bien...", h - 40, YELLOW, 0.6)
    elif st.state == "ESTABILIZANDO":
        put(frame, f"Mantente quieto: guardando... {st.progress * 100:.0f}%",
            h - 40, YELLOW, 0.6)
    elif st.state == "REFRESCO_PENDIENTE":
        put(frame, "Actualizacion pendiente: esperando deteccion correcta",
            h - 40, YELLOW, 0.6)
    if st.profile is not None:
        p = st.profile
        put(frame, f"Perfil: {p.forma['label']} | {p.somatotipo['label']} | "
                   f"piel {p.piel['label']}", h - 64, GREEN, 0.6)
        put(frame, f"Expira en {fmt_time(st.expires_in)}  |  "
                   f"proximo guardado en {fmt_time(st.refresh_in)}",
            h - 16, WHITE, 0.5)
    put(frame, f"ciclo IA: {cycle_ms:.0f} ms", 100, WHITE, 0.45)


def on_save(profile: Profile) -> None:
    print("\n[PERFIL GUARDADO]")
    print(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))


def main() -> int:
    cfg = parse_args(Config())

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara (indice {cfg.camera_index}).")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Cargando las 3 IAs (la primera vez puede tardar)...")
    analyzers = [
        BodyShapeAnalyzer(cfg.body_smoothing_alpha),
        SomatotypeAnalyzer(SOMATO_WEIGHTS, cfg.min_somato_conf),
        SkinToneAnalyzer(cfg.min_skin_accuracy),
    ]
    manager = ProfileManager(
        ttl_seconds=cfg.ttl_seconds,
        refresh_seconds=cfg.refresh_seconds,
        stable_seconds=cfg.stable_seconds,
        error_grace_seconds=cfg.error_grace_seconds,
        on_save=on_save,
    )
    pipeline = AnalysisPipeline(analyzers, manager, cfg.analysis_period)
    pipeline.start()

    print("Camara iniciada. q = salir | r = reiniciar | p = imprimir perfil")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer el frame de la camara.")
                break
            frame = cv2.flip(frame, 1)

            if pipeline.wants_frame():
                pipeline.submit(frame.copy())

            draw_hud(frame, manager.status(), pipeline.cycle_ms)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                manager.clear()
                print("Perfil descartado.")
            if key == ord("p"):
                prof = manager.get_profile()
                print(json.dumps(prof.to_dict(), indent=2, ensure_ascii=False)
                      if prof else "Aun no hay perfil guardado.")
    finally:
        pipeline.shutdown()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())