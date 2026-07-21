import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = np.expand_dims(x_train.astype("float32") / 255.0, axis=-1)
x_test = np.expand_dims(x_test.astype("float32") / 255.0, axis=-1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("Iniciando treinamento...")
history = model.fit(
    x_train,
    y_train,
    epochs=1,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

val_acc = history.history["val_accuracy"][-1]
_, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f"Val Acc: {val_acc:.4f}")
print(f"Test Acc: {test_acc:.4f}")

model.save("model.h5")
print("Modelo salvo em model.h5")