import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator


# Rutas de las carpetas test, train y valid
train_dir = 'train' 
valid_dir = 'valid'
test_dir = 'test'

# Creacion de las clases 
class_names = ['knife', 'no_risk','weapon']

# Definir la normalización de los datos
train_data_gen = ImageDataGenerator(
    rotation_range=90,  # Rango de rotación aleatoria de hasta 90 grados
    horizontal_flip=True  # Reflejo horizontal aleatorio
)
valid_data_gen = ImageDataGenerator(
    rotation_range=90,  # Rango de rotación aleatoria de hasta 90 grados
    horizontal_flip=True  # Reflejo horizontal aleatorio
)
test_data_gen = ImageDataGenerator(
    rotation_range=90,  # Rango de rotación aleatoria de hasta 90 grados
    horizontal_flip=True  # Reflejo horizontal aleatorio
)

# Generar los datos de entrenamiento
train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse'
)

valid_generator = valid_data_gen.flow_from_directory(
    valid_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse'
)

test_generator = test_data_gen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse'
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
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo con los generadores
history_model = model.fit(train_generator, validation_data=valid_generator, epochs=100)

# Guardar el modelo
model.save('CNN_Modelo.h5')

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Accuracy: {test_accuracy}')
