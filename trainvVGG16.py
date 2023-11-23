import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def build_model(bottom_model, classes):
    model = bottom_model.layers[-2].output
    model = GlobalAveragePooling2D()(model)
    model = Dense(classes, activation='softmax', name="Capa_Salida")(model)

    return model

# Rutas de las carpetas test, train y valid
train_dir = 'dataset_3/train'
valid_dir = 'dataset_3/valid'
test_dir = 'dataset_3/test'
BATCH_SIZE = 32

# Creacion de las clases 
class_names = ['no_risk', 'risk']

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

head = build_model(vgg, len(class_names))

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

# Graficar la precisión de entrenamiento y validación por época
plt.plot(history_model.history['accuracy'])
plt.plot(history_model.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.show()

# Graficar la pérdida de entrenamiento y validación por época
plt.plot(history_model.history['loss'])
plt.plot(history_model.history['val_loss'])
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.show()

# Guardar el modelo
model.save('CNN_Modelo-VGG16.h5')

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Accuracy: {test_accuracy}')

# Obtener las predicciones del modelo en el conjunto de datos de prueba
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Obtener las etiquetas reales del conjunto de datos de prueba
true_classes = test_generator.classes

# Calcular la matriz de confusión
confusion_matrix = confusion_matrix(true_classes, predicted_classes)

# Imprimir la matriz de confusión
print("Confusion Matrix")
print(confusion_matrix)

# Obtener un informe de clasificación
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Obtener las probabilidades de predicción del modelo
predicted_probabilities = model.predict(test_generator)

# Calcular la curva ROC para cada clase
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(test_generator.classes == i, predicted_probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Dibuja las curvas ROC para cada clase
plt.figure(figsize=(8, 6))

for i in range(len(class_names)):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve (area = {roc_auc[i]:.2f}) for {class_names[i]}')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
