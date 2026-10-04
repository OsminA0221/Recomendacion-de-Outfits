import cv2
import stone
import os

TEMP_IMAGE = "temp_frame.jpg"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Cámara iniciada. Presiona Q para salir.")

frame_count = 0
skin_tone = "Detectando..."
tone_label = ""
accuracy = ""

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo obtener un frame de la cámara.")
        break

    # Efecto espejo
    frame = cv2.flip(frame, 1)

    frame_count += 1

    # Analizar cada 10 frames
    if frame_count % 10 == 0:
        cv2.imwrite(TEMP_IMAGE, frame)

        try:
            result = stone.process(
                TEMP_IMAGE,
                image_type="color"
            )

            faces = result.get("faces", [])

            if faces:
                face = faces[0]

                skin_tone = face.get("skin_tone", "N/A")
                tone_label = face.get("tone_label", "N/A")
                accuracy = face.get("accuracy", "N/A")

            else:
                skin_tone = "No detectado"
                tone_label = ""
                accuracy = ""

        except Exception as e:
            print("Error al analizar imagen:", e)

    # Mostrar información en pantalla
    cv2.putText(
        frame,
        f"Tono: {skin_tone}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Etiqueta: {tone_label}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Precision: {accuracy}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Deteccion de Tonalidad de Piel", frame)

    # Q para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if os.path.exists(TEMP_IMAGE):
    os.remove(TEMP_IMAGE)

print("Camara cerrada.")