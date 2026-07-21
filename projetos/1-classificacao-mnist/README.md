# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Fabricio Melo

### 1️⃣ Resumo da Arquitetura do Modelo
* **Estrutura:** Rede Neural Convolucional (CNN) com 3 blocos de convolução (`Conv2D`).
* **Componentes:** Usa `BatchNormalization` após cada convolução para acelerar o aprendizado e uma camada de `Dropout` (0.3) antes da saída para evitar o sobreajuste.
* **Treinamento:** Utiliza 10% dos dados para validação e o mecanismo `EarlyStopping` (paciência de 3 épocas) para parar o treino assim que o modelo para de evoluir.

### 2️⃣ Bibliotecas Utilizadas
* `TensorFlow / Keras` (Versão 2.15+) — Construção, treino e conversão do modelo.
* `NumPy` — Manipulação das matrizes de imagem.
* `Matplotlib` — Geração e salvamento do gráfico de teste.

### 3️⃣ Técnica de Otimização do Modelo
Foi utilizada a técnica de **Quantização de Escopo Dinâmico (*Dynamic Range Quantization*)** através do `TFLiteConverter`. Essa técnica converte os pesos do modelo de ponto flutuante (32 bits) para formato inteiro (8 bits). Isso reduz o tamanho do arquivo e faz o modelo rodar muito mais rápido na CPU de sistemas embarcados.

### 4️⃣ Resultados Obtidos
* **Acurácia de Validação:** 99.00%
* **Tamanho do `model.h5`:** 1169.3 KB
* **Tamanho do `model.tflite`:** 104.8 KB
* **Redução de Tamanho:** 91.0% de economia de espaço.

### 5️⃣ Comentários Adicionais (Opcional)
O código foi desenvolvido de forma linear e simplificada para garantir uma execução rápida e sem erros dentro do Dev Container. A inserção do `BatchNormalization` ajudou o modelo a atingir a acurácia máxima em apenas 5 épocas, mostrando que a rede ficou leve e eficiente para rodar direto em CPU.

### 6️⃣ Exemplo de Inferência
Abaixo consta a saída real do terminal obtida ao executar o script `run_inference.py` utilizando o interpretador do TensorFlow Lite sobre as primeiras amostras do conjunto de teste isolado:

```text
Executando validacao automatica sobre 5 amostras reais (model.tflite):

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```

O modelo demonstrou alta precisão ao classificar corretamente o dígito manuscrito sorteado, comprovando a eficácia do treinamento.

