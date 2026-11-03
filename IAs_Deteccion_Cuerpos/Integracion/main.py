from __future__ import annotations

import argparse
import json
import sys
import threading
from dataclasses import replace
from pathlib import Path

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

# Sube un nivel desde 'Integracion' hacia 'IAs_Deteccion_Cuerpos'
BASE_DIR = Path(__file__).resolve().parent.parent

# Ruta hacia la carpeta 'src' que contiene 'stone'
RUTA_SRC_STONE = BASE_DIR / "Tonalidades_Piel" / "src"

if str(RUTA_SRC_STONE) not in sys.path:
    sys.path.insert(0, str(RUTA_SRC_STONE))

# Raiz del proyecto (contiene data/ y Recomendaciones/) para poder importar
# el paquete Recomendaciones.
RAIZ = BASE_DIR.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from Recomendaciones import COLORES_PREFERIDOS, Recomendador, perfil_desde_dict   # noqa: E402
from Recomendaciones.visual import crear_collage              # noqa: E402

WINDOW = "Outfit AI - 3 IAs en paralelo"
WINDOW_OUTFITS = "Outfits recomendados"
ESTILOS = [None, "casual", "elegante", "formal", "deportivo", "streetwear"]
OCASIONES = [None, "casual", "elegante", "formal", "playa", "frio", "deportivo", "streetwear"]
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, YELLOW, WHITE, CYAN = (0, 220, 0), (0, 0, 255), (0, 215, 255), (255, 255, 255), (255, 200, 0)

LABELS = {"forma": "Forma", "somatotipo": "Somatotipo", "piel": "Piel"}


def parse_args(cfg: Config) -> Config:
    p = argparse.ArgumentParser(description="3 IAs en paralelo con camara en vivo")
    p.add_argument("--camera", type=int, default=cfg.camera_index)
    p.add_argument("--gender", type=str, default=cfg.gender,
                   choices=["Hombre", "Mujer", "hombre", "mujer", "h", "m"],
                   help="Genero del usuario: Hombre o Mujer (default: Mujer)")
    p.add_argument("--stable", type=float, default=cfg.stable_seconds,
                   help="segundos de deteccion estable antes de guardar (2 o 5)")
    p.add_argument("--refresh", type=float, default=cfg.refresh_seconds,
                   help="cada cuantos segundos se vuelve a guardar")
    p.add_argument("--ttl", type=float, default=cfg.ttl_seconds,
                   help="segundos que vive el perfil guardado")
    a = p.parse_args()
    g = "Hombre" if str(a.gender).strip().lower() in ("h", "hombre", "man", "male") else "Mujer"
    return replace(cfg, camera_index=a.camera, gender=g, stable_seconds=a.stable,
                   refresh_seconds=a.refresh, ttl_seconds=a.ttl)


def fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def put(frame, text, y, color=WHITE, scale=0.55):
    cv2.putText(frame, text, (12, y), FONT, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (12, y), FONT, scale, color, 1, cv2.LINE_AA)


def short(data: dict, name: str) -> str:
    if name == "forma":
        g = data.get("gender", "")
        g_tag = f"[{g}] " if g else ""
        return f"{g_tag}{data['label']} (h/c {data['shoulder_to_hip']}, c/c {data['waist_to_hip']})"
    if name == "somatotipo":
        return f"{data['label']} {data['confidence'] * 100:.0f}%"
    return f"{data['label']} {data['skin_tone']} prec {data['accuracy']:.0f}"


def draw_hud(frame, st, cycle_ms, estado):
    # Indicador de Género activo y controles rápidos
    put(frame, f"[ GENERO: {st.gender.upper()} ]  (Presiona H: Hombre | M: Mujer | G: Alternar)", 26, CYAN, 0.58)

    estilo = ESTILOS[estado["estilo_idx"]] or "todos"
    ocasion = OCASIONES[estado["ocasion_idx"]] or "todos"
    color = COLORES_PREFERIDOS[estado["color_idx"]].capitalize()
    put(frame, f"[ ESTILO: {estilo.upper()} ]  (E para cambiar)", 156, CYAN, 0.55)
    put(frame, f"[ OCASION: {ocasion.upper()} ]  (X para cambiar)", 180, CYAN, 0.55)
    put(frame, f"[ COLOR: {color.upper()} ]  (C para cambiar)", 204, CYAN, 0.55)

    y = 54
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

    r_forma = st.last_results.get("forma")
    if r_forma and r_forma.ok and "edges" in r_forma.data:
        rows = (("hombros", (255, 0, 0)), ("cintura", (0, 255, 255)),
                ("cadera", (255, 0, 255)))
        for key, color in rows:
            e = r_forma.data["edges"].get(key)
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
        put(frame, f"Perfil [{p.genero}]: {p.forma['label']} | {p.somatotipo['label']} | "
                   f"piel {p.piel['label']}", h - 64, GREEN, 0.6)
        put(frame, f"Expira en {fmt_time(st.expires_in)}  |  "
                   f"proximo guardado en {fmt_time(st.refresh_in)}",
            h - 16, WHITE, 0.5)
    put(frame, f"ciclo IA: {cycle_ms:.0f} ms", 130, WHITE, 0.45)


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

    print("Cargando recomendador...")
    rec = Recomendador()
    lock = threading.Lock()
    estado = {
        "outfits": None, "perfil_dict": None, "estilo_idx": 0,
        "ocasion_idx": 0, "color_idx": 0, "nuevo": False,
    }

    def recalcular(perfil_dict: dict) -> None:
        """Perfil de las 3 IAs -> outfits; el hilo principal los dibuja."""
        try:
            outfits = rec.recomendar_outfits(
                perfil_desde_dict(perfil_dict), n=4,
                estilo=ESTILOS[estado["estilo_idx"]],
                ocasion=OCASIONES[estado["ocasion_idx"]],
                color_preferido=COLORES_PREFERIDOS[estado["color_idx"]])
        except Exception as exc:   # un error aqui no debe tumbar el analisis
            print(f"[RECOMENDADOR] error: {type(exc).__name__}: {exc}")
            return
        with lock:
            estado.update(perfil_dict=perfil_dict, outfits=outfits, nuevo=True)

    def on_save_y_recomendar(profile: Profile) -> None:
        on_save(profile)                      # imprime el JSON como antes
        recalcular(profile.to_dict())

    print("Cargando las 3 IAs (la primera vez puede tardar)...")
    body_analyzer = BodyShapeAnalyzer(cfg.body_smoothing_alpha, gender=cfg.gender)
    analyzers = [
        body_analyzer,
        SomatotypeAnalyzer(SOMATO_WEIGHTS, cfg.min_somato_conf),
        SkinToneAnalyzer(cfg.min_skin_accuracy),
    ]
    manager = ProfileManager(
        ttl_seconds=cfg.ttl_seconds,
        refresh_seconds=cfg.refresh_seconds,
        stable_seconds=cfg.stable_seconds,
        error_grace_seconds=cfg.error_grace_seconds,
        initial_gender=cfg.gender,
        on_save=on_save_y_recomendar,
    )
    pipeline = AnalysisPipeline(analyzers, manager, cfg.analysis_period)
    pipeline.start()

    print("Camara iniciada.")
    print("  q = salir | r = reiniciar | p = imprimir perfil")
    print("  h = seleccionar Hombre | m = seleccionar Mujer | g = alternar genero")
    print("  e = cambiar estilo (todos/casual/elegante/formal/deportivo/streetwear)")
    print("  x = cambiar ocasion (todos/casual/elegante/formal/playa/frio/deportivo/streetwear)")
    print("  c = cambiar color preferido")
    print("  o = recalcular outfits")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer el frame de la camara.")
                break
            frame = cv2.flip(frame, 1)

            if pipeline.wants_frame():
                pipeline.submit(frame.copy())

            draw_hud(frame, manager.status(), pipeline.cycle_ms, estado)
            estilo = ESTILOS[estado["estilo_idx"]] or "todos"
            ocasion = OCASIONES[estado["ocasion_idx"]] or "todos"
            color = COLORES_PREFERIDOS[estado["color_idx"]]
            cv2.setWindowTitle(
                WINDOW,
                f"{WINDOW} | Estilo: {estilo} | Ocasion: {ocasion} | Color: {color}",
            )
            cv2.imshow(WINDOW, frame)
            # Los outfits se dibujan aqui (cv2.imshow solo en el hilo principal)
            with lock:
                hay_nuevo, outfits = estado["nuevo"], estado["outfits"]
                estado["nuevo"] = False
            if hay_nuevo:
                cv2.imshow(WINDOW_OUTFITS, crear_collage(outfits))
                cv2.setWindowTitle(
                    WINDOW_OUTFITS,
                    f"{WINDOW_OUTFITS} | Estilo: {estilo} | Ocasion: {ocasion} | Color: {color}",
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                manager.clear()
                with lock:
                    estado["perfil_dict"] = None
                print("Perfil descartado.")
            if key == ord("p"):
                prof = manager.get_profile()
                print(json.dumps(prof.to_dict(), indent=2, ensure_ascii=False)
                      if prof else "Aun no hay perfil guardado.")
            if key in (ord("h"), ord("H")):
                new_g = "Hombre"
                manager.set_gender(new_g)
                body_analyzer.set_gender(new_g)
                with lock:
                    estado["perfil_dict"] = None
                print(f"[GENERO] Cambiado a: {new_g}")
            if key in (ord("m"), ord("M")):
                new_g = "Mujer"
                manager.set_gender(new_g)
                body_analyzer.set_gender(new_g)
                with lock:
                    estado["perfil_dict"] = None
                print(f"[GENERO] Cambiado a: {new_g}")
            if key in (ord("g"), ord("G")):
                new_g = "Hombre" if manager.gender == "Mujer" else "Mujer"
                manager.set_gender(new_g)
                body_analyzer.set_gender(new_g)
                with lock:
                    estado["perfil_dict"] = None
                print(f"[GENERO] Alternado a: {new_g}")
            if key in (ord("e"), ord("E")):
                estado["estilo_idx"] = (estado["estilo_idx"] + 1) % len(ESTILOS)
                print(f"[ESTILO] {ESTILOS[estado['estilo_idx']] or 'todos'}")
                if estado["perfil_dict"]:
                    recalcular(estado["perfil_dict"])
            if key in (ord("x"), ord("X")):
                estado["ocasion_idx"] = (estado["ocasion_idx"] + 1) % len(OCASIONES)
                print(f"[OCASION] {OCASIONES[estado['ocasion_idx']] or 'todos'}")
                if estado["perfil_dict"]:
                    recalcular(estado["perfil_dict"])
            if key in (ord("c"), ord("C")):
                estado["color_idx"] = (estado["color_idx"] + 1) % len(COLORES_PREFERIDOS)
                print(f"[COLOR] {COLORES_PREFERIDOS[estado['color_idx']]}")
                if estado["perfil_dict"]:
                    recalcular(estado["perfil_dict"])
            if key in (ord("o"), ord("O")):
                prof = manager.get_profile()
                if prof:
                    recalcular(prof.to_dict())
                else:
                    print("Aun no hay perfil guardado.")
    finally:
        pipeline.shutdown()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())