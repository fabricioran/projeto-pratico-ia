#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Projeto 1 — Inferência com o Modelo Otimizado (model.tflite)
# ---------------------------------------------------------------------------

def main():
    # 1. Localizar e carregar estritamente o artefato de edge usando tf.lite.Interpreter
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_tflite = os.path.join(diretorio_atual, "model.tflite")
    
    interpretador = tf.lite.Interpreter(model_path=caminho_tflite)
    interpretador.allocate_tensors()
    
    detalhes_entrada = interpretador.get_input_details()
    detalhes_saida = interpretador.get_output_details()

    # Carregar dados oficiais de teste para validacao pontual
    (_, _), (imagens_teste, gabarito_teste) = tf.keras.datasets.mnist.load_data()
    imagens_teste = imagens_teste.astype("float32") / 255.0
    imagens_teste = np.expand_dims(imagens_teste, axis=-1)

    print("Executando validacao automatica sobre 5 amostras reais (model.tflite):\n")
    
    # 2. Iteracao obrigatoria de amostragem
    for i in range(5):
        vetor_entrada = np.expand_dims(imagens_teste[i], axis=0)
        
        interpretador.set_tensor(detalhes_entrada[0]["index"], vetor_entrada)
        interpretador.invoke()
        
        vetor_saida = interpretador.get_tensor(detalhes_saida[0]["index"])[0]
        numero_predito = int(np.argmax(vetor_saida))
        numero_real = int(gabarito_teste[i])
        
        # 3. Exibicao exigida: classe predita vs classe real
        print(f"Amostra {i + 1}: predito={numero_predito} | real={numero_real}")

if __name__ == "__main__":
    main()
