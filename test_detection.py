import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Cargar el modelo entrenado
model = tf.keras.models.load_model('Entrenamientos/CNN_Modelo7.h5')

# Abrir el archivo .mp4
video_path = 'Videos/Arma3.mp4'
cap = cv2.VideoCapture(video_path)

# Crear el generador de datos para normalización
data_gen = ImageDataGenerator(
    rotation_range=90,  # Rango de rotación aleatoria de hasta 90 grados
    horizontal_flip=True  # Reflejo horizontal aleatorio
)

# Inicializar la ventana para mostrar el video
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# Procesar el video fotograma por fotograma
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar el fotograma a 150x150 píxeles
    frame = cv2.resize(frame, (150, 150))

    # Normalizar la imagen
    frame = data_gen.standardize(np.array([frame]))

    # Realizar la inferencia con el modelo
    predictions = model.predict(frame)

    # Interpretar las predicciones
    class_index = np.argmax(predictions)
    class_labels = ['risk', 'no_risk' ]
    class_label = f"Clase predicha: {class_labels[class_index]}"

    # Dibujar el cuadro de texto en el fotograma
    cv2.putText(frame[0], class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)

    cv2.imshow('Video', frame[0])  # Mostrar el fotograma

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
