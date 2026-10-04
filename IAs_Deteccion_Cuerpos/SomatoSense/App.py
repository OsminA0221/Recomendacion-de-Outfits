from flask import Flask, render_template, request, send_from_directory, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import uuid
import cv2
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static'  # Manténlo como 'static' para Flask
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Cargar tu modelo entrenado
model = YOLO("best.pt")

# -----------------------------------------------------------
# DETECCIÓN GENERAL
# -----------------------------------------------------------
def detectar_objetos(imagen_path):
    results = model.predict(source=imagen_path, save=False)
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            detections.append({'name': name, 'confidence': round(conf * 100, 2)})

    result_name = f"result_{uuid.uuid4().hex[:6]}.jpg"
    save_path = os.path.join(RESULT_FOLDER, result_name)
    
    # DEBUG: Verifica la ruta
    print(f"DEBUG - Ruta absoluta de save_path: {os.path.abspath(save_path)}")
    
    # Guarda la imagen usando cv2 en lugar de results.save()
    try:
        annotated_frame = results[0].plot()
        success = cv2.imwrite(save_path, annotated_frame)
        if success:
            print(f"DEBUG - Imagen guardada con éxito en: {save_path}")
            print(f"DEBUG - ¿Existe el archivo?: {os.path.exists(save_path)}")
        else:
            print("DEBUG - Error: cv2.imwrite() falló")
    except Exception as e:
        print(f"DEBUG - Error al guardar: {e}")
        # Fallback: intenta con el método original
        try:
            results[0].save(filename=save_path)
            print(f"DEBUG - Imagen guardada con método alternativo")
        except Exception as e2:
            print(f"DEBUG - Error con método alternativo: {e2}")

    return result_name, detections

# -----------------------------------------------------------
# RUTA PARA SERVIR ARCHIVOS DESDE STATIC
# -----------------------------------------------------------
# @app.route('/static/<filename>')
# def serve_static(filename):
#     return send_from_directory('static', filename)

# -----------------------------------------------------------
# RUTA PRINCIPAL
# -----------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            result_image, detections = detectar_objetos(filepath)
            os.remove(filepath)

            return render_template('index.html', result_image=result_image, detections=detections)

    return render_template('index.html')

# -----------------------------------------------------------
# STREAMING EN TIEMPO REAL
# -----------------------------------------------------------
def gen_frames():
    cap = cv2.VideoCapture(0)
    last_detection_time = 0
    cached_frame_with_detections = None
    cache_valid_until = 0
    detection_interval = 2
    cache_duration = 5
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        
        if current_time - last_detection_time >= detection_interval:
            results = model(frame, conf=0.1)
            cached_frame_with_detections = results[0].plot()
            last_detection_time = current_time
            cache_valid_until = current_time + cache_duration
        
        if cached_frame_with_detections is not None and current_time <= cache_valid_until:
            output_frame = cached_frame_with_detections
        else:
            output_frame = frame
        
        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------------------------------------
# EJECUTAR
# -----------------------------------------------------------
if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)

