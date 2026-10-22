import sys
import cv2

from src.pose_detector import PoseDetector
from src.body_measurements import (
    compute_proportions,
    BodyMeasurementSmoother
)
from src.body_shape_classifier import classify_body_shape
from src.overlay import (
    draw_results,
    draw_no_detection_warning
)


CAMERA_INDEX = 0

WINDOW_NAME = (
    "Outfit AI - Primera fase: Deteccion de forma de cuerpo"
)


def main() -> int:

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        print(
            f"No se pudo abrir la camara "
            f"(indice {CAMERA_INDEX})."
        )

        print(
            "Verifica que este conectada "
            "y no este en uso por otra app."
        )

        return 1

    detector = PoseDetector()

    measurement_smoother = (
        BodyMeasurementSmoother.create(
            alpha=0.20
        )
    )

    current_gender = "Mujer"

    print(
        "Camara iniciada.\n"
        "  q = salir\n"
        "  h = cambiar a Hombre\n"
        "  m = cambiar a Mujer\n"
        "  g = alternar genero"
    )

    try:

        while True:

            ok, frame = cap.read()

            if not ok:

                print(
                    "No se pudo leer "
                    "el frame de la camara."
                )

                break

            frame = cv2.flip(
                frame,
                1
            )
            
            alto_imagen, ancho_imagen, _ = frame.shape

            landmarks = detector.process(
                frame
            )

            frame = detector.draw_landmarks(
                frame
            )

            if (
                landmarks is not None
                and landmarks.visibility_ok
            ):

                proportions = compute_proportions(
                    landmarks,
                    detector=detector,
                    smoother=measurement_smoother
                )

                shape_result = (
                    classify_body_shape(
                        proportions,
                        gender=current_gender
                    )
                )

                frame = draw_results(
                    frame,
                    proportions,
                    shape_result
                )

            else:

                # Si perdemos la persona,
                # reiniciamos el filtro para evitar
                # conservar valores antiguos.
                measurement_smoother.reset()

                frame = (
                    draw_no_detection_warning(
                        frame
                    )
                )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("h"), ord("H")):
                current_gender = "Hombre"
                print(f"[GENERO] Cambiado a: {current_gender}")
            elif key in (ord("m"), ord("M")):
                current_gender = "Mujer"
                print(f"[GENERO] Cambiado a: {current_gender}")
            elif key in (ord("g"), ord("G")):
                current_gender = "Hombre" if current_gender == "Mujer" else "Mujer"
                print(f"[GENERO] Alternado a: {current_gender}")

    finally:

        cap.release()

        detector.close()

        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())