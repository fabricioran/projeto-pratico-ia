import os
import tensorflow as tf

# Carrega o modelo com tratamento de compatibilidade
try:
    import tf_keras as keras
    model = keras.models.load_model("model.h5")
except Exception:
    try:
        model = tf.keras.saving.legacy.load_model("model.h5")
    except Exception:
        model = tf.keras.models.load_model("model.h5")

# 1. Converter para TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 2. Aplicar otimização explícita (Dynamic Range Quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# 3. Salvar como model.tflite
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo otimizado salvo em 'model.tflite' com sucesso!")