import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model('CNN_Modelo.h5')

# Abrir el archivo .mp4 
video_path = 'Arma.mp4'
cap = cv2.VideoCapture(video_path)

# Procesar el video fotograma por fotograma
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar la inferencia con el modelo
    input_data = np.expand_dims(frame, axis=0)  # Agregar una dimensión adicional para el lote
    predictions = model.predict(input_data)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
