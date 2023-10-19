import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# Rutas de las carpetas test, train y valid
train_dir = 'train\images' 
valid_dir = 'valid\images'
test_dir = 'test\images'

# Definir la normalización de los datos
train_dir = ImageDataGenerator()
valid_dir = ImageDataGenerator()
test_dir = ImageDataGenerator()

class_names = ['pistol', 'knife', 'free risk']

# Generar los datos de entrenamiento
train_generator = train_dir.flow_from_directory(
    train_dir,  # Usa la ruta completa al directorio de entrenamiento
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    # classes=class_names
)

valid_generator = valid_dir.flow_from_directory(
    valid_dir,  # Usa la ruta completa al directorio de validación
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    # classes=class_names
)

test_generator = test_dir.flow_from_directory(
    test_dir,  # Usa la ruta completa al directorio de prueba
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    # classes=class_names
)


# Arquitectura de la CNN
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo con los generadores
history_model = model.fit(train_generator, validation_data=valid_generator, epochs=20)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Accuracy: {test_accuracy}')
