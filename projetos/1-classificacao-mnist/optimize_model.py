#!/usr/bin/env python3
import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# 1. Carrega o arquivo gerado pelo treino
modelo = tf.keras.models.load_model("model.h5")

# 2. Configura conversor para formato leve (TFLite) usando o comando correto
conversor = tf.lite.TFLiteConverter.from_keras_model(modelo)

# 3. Aplica a quantizacao para rodar leve em sistemas embarcados
conversor.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Converte e salva o arquivo otimizado com o nome exigido
modelo_tflite = conversor.convert()
nome_saida = "model.tflite"

with open(nome_saida, "wb") as f:
    f.write(modelo_tflite)

# 5. Imprime o relatorio de tamanhos no terminal
tam_h5 = os.path.getsize("model.h5") / 1024
tam_tflite = os.path.getsize(nome_saida) / 1024

print("\n--- Conversao Concluida ---")
print(f"Modelo antigo (.h5): {tam_h5:.1f} KB")
print(f"Modelo novo (.tflite): {tam_tflite:.1f} KB")
print(f"Diferenca: {((1 - (tam_tflite / tam_h5)) * 100):.1f}% menor")
