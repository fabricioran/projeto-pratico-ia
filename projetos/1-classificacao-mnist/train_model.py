import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 1. Carregar dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalizar escala para [0, 1] e ajustar dimensões para (28, 28, 1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Mudar alvos para matrizes categóricas (10 classes)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 3. Construção da CNN (Recomendado para Keras 3: camada Input separada)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    
    # Bloco 1
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Bloco 2
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Bloco 3
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    
    # Camadas Densas / Classificador
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compilação do modelo
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks: EarlyStopping monitorando a perda de validação
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# 4. Treinamento (epochs=1 para validação rápida no CI/CD)
print("Iniciando o treinamento do modelo...")
historico = model.fit(
    x_train,
    y_train,
    epochs=1,  # Reduzido para passar rápido no GitHub Actions
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# 5. Exibição da métrica no terminal
acc_val = historico.history["val_accuracy"][-1]
print(f"\nAcuracia final de validacao: {acc_val:.4f}")

_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acuracia obtida no teste: {test_acc:.4f}\n")

# 6. Salvar modelo em model.h5 (Forma universal compatível com Keras 3/TF 2.x)
model.save("model.h5")

print("Modelo salvo em 'model.h5' com sucesso!")