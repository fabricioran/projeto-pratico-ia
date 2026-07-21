## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização applied
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Fabricio Melo

### 1️⃣ Resumo da Arquitetura do Modelo
* **Estrutura:** Rede Neural Convolucional (CNN) simples composta por 3 blocos de extração de características.
* **Componentes:** Cada bloco utiliza `Conv2D` com ativação ReLU, seguido por `BatchNormalization` para estabilização das distribuições intermediárias de ativamentos e `MaxPooling2D` para redução de dimensionalidade espacial.
* **Classificação:** Camada `Flatten`, seguida de camada densa com 64 neurônios, `Dropout` (0.3) para prevenção de overfitting e camada final `Dense(10)` com softmax.
* **Justificativa Técnica de Hiperparâmetro:** Optou-se pela escolha de **batch_size = 64** para equilibrar a estabilidade da estimativa do gradiente estocástico com a eficiência de uso do cache de memória da CPU durante o processamento no ambiente de treino. Adicionalmente, a taxa de **Dropout de 0.3 (30%)** foi definida para evitar a co-adaptação excessiva dos neurônios nas camadas densas sem penalizar a capacidade de representação do modelo.

### 2️⃣ Bibliotecas Utilizadas
* `TensorFlow / Keras` (Versão 2.15+) — Construção, compilação, salvamento do modelo e conversão TFLite.
* `NumPy` (Versão 1.24+) — Processamento matricial, normalização e manipulação de arrays do dataset.

### 3️⃣ Técnica de Otimização do Modelo
Utilizou-se a técnica de **Quantização Dinâmica de Intervalo (*Dynamic Range Quantization*)** através do `tf.lite.TFLiteConverter` com `optimizations = [tf.lite.Optimize.DEFAULT]`. Durante a conversão, os pesos do modelo são quantizados de ponto flutuante de precisão simples (FP32) para inteiros de 8 bits (INT8), permitindo footprint reduzido e execução otimizada em hardware embarcado (Edge AI).

### 4️⃣ Resultados Obtidos
* **Acurácia de Validação:** > 98.9%
* **Tamanho do `model.h5`:** 1173.5 KB
* **Tamanho do `model.tflite`:** 104.8 KB
* **Redução de Tamanho:** 91.1% de economia de armazenamento.

### 5️⃣ Comentários Adicionais (Opcional)
Durante o desenvolvimento, a principal decisão e limitação técnica tratada foi a adequação às atualizações do ecossistema **Keras 3 / TensorFlow 2.x**. A utilização de chamadas legadas de salvamento gerava erros de herança de grafos (`_is_graph_network`) e avisos de depreciação. O código foi totalmente refatorado para utilizar a camada explícita `Input(shape=(28, 28, 1))` e a API nativa `model.save()`. Além disso, para evitar estouro de tempo (*timeout*) nos runners sem GPU do GitHub Actions, o treinamento foi ajustado com suporte ao callback `EarlyStopping` e número reduzido de épocas operacionais para validação contínua e rápida em esteiras de CI/CD.

### 6️⃣ Exemplo de Inferência
Saída real de execução obtida pelo script `run_inference.py` utilizando o interpretador do TensorFlow Lite sobre 5 amostras isoladas do conjunto de teste:

```text
Executando inferência em 5 amostras:

Amostra 1: predito = 7 | real = 7
Amostra 2: predito = 2 | real = 2
Amostra 3: predito = 1 | real = 1
Amostra 4: predito = 0 | real = 0
Amostra 5: predito = 4 | real = 4