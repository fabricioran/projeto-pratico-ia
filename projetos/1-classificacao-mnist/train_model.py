import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------

# 1. Pegar dados do MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Arrumar escala para e ajustar dimensões
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Mudar alvos para matrizes categóricas
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 4. Criar a rede sequencial com sintaxe estrita e posicional (Compatibilidade Total)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

parada = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

print("Iniciando o ajuste dos pesos do modelo...")
historico = model.fit(
    x_train, 
    y_train, 
    epochs=10, 
    batch_size=64, 
    validation_split=0.1, 
    callbacks=[parada]
)

acc_final = historico.history["val_accuracy"][-1]
print(f"\nAcuracia final de validacao: {acc_final:.4f}")

_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acuracia obtida no teste: {test_acc:.4f}\n")

# 7. Gravar salvando apenas os pesos e arquitetura de forma limpa
model.save_weights("model.h5")
print("Modelo guardado com sucesso!")

idx = np.random.randint(0, len(x_test))
pred = model.predict(np.expand_dims(x_test[idx], axis=0), verbose=0)

plt.figure(figsize=(4, 4))
plt.imshow(x_test[idx].squeeze(), cmap="gray")
plt.title(f"Real: {np.argmax(y_test[idx])} | Modelo: {np.argmax(pred)}")
plt.axis("off")
plt.savefig("exemplo_previsao.png")
