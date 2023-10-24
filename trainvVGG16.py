import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model

# Rutas de las carpetas test, train y valid
train_dir = 'train' 
valid_dir = 'valid'
test_dir = 'test'
BATCH_SIZE = 32

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
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode='categorical', shuffle=True
)

valid_generator = valid_data_gen.flow_from_directory(
    valid_dir,
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode='categorical', shuffle=True
)

test_generator = test_data_gen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode='categorical', shuffle=True
)

vgg = tf.keras.applications.VGG16(input_shape=(150,150,3),include_top=False,weights="imagenet")

vgg.summary()

def build_model(bottom_model, classes):
    model = bottom_model.layers[-2].output
    model = GlobalAveragePooling2D()(model)
    model = Dense(classes, activation='softmax', name="Capa_Salida")(model)

    return model

head = build_model(vgg, 2)

model = Model(inputs= vgg.input, outputs = head)
model.summary()


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo con los generadores
history_model = model.fit(train_generator, 
                          validation_data=valid_generator, 
                          epochs=50, 
                          steps_per_epoch=train_generator.n//BATCH_SIZE,
                          validation_steps=valid_generator.n//BATCH_SIZE)

# Guardar el modelo
model.save('CNN_Modelo-VGG16.h5')

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Accuracy: {test_accuracy}')
