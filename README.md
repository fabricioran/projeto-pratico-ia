# Classificação Computacional de Padrões e Otimização em Arquiteturas Edge (MNIST)

## 📜 Contexto Histórico e a Gênese da Visão Computacional Moderna

Na intersecção entre a neurobiologia e a computação gráfica da década de 1980, o reconhecimento automático de caracteres manuscritos emergiu como um dos desafios mais complexos da inteligência artificial primitiva. Sistemas postais automáticos e instituições bancárias dependiam crucialmente da digitalização de envelopes e cheques, esbarrando na variabilidade morfológica da caligrafia humana.

Para balizar o desenvolvimento de algoritmos de reconhecimento de padrões, o *National Institute of Standards and Technology* (NIST) coletou originalmente dois bancos de dados complementares nos Estados Unidos: o *Special Database 1* (composto por dígitos escritos por estudantes do ensino médio) e o *Special Database 3* (composto por amostras coletadas de funcionários do escritório do censo americano). Embora ricos, os conjuntos sofriam de disparidades gritantes de distribuição e estilo de traço, dificultando o treinamento estável de classificadores estatísticos.

No final da década de 1990, pesquisadores liderados por Yann LeCun, Corinna Cortes e Christopher J.C. Burges promoveram uma reestruturação profunda nesses dados, criando a variante **MNIST** (*Modified MNIST*). O processo envolveu:
1. **Fusão e Embaralhamento:** Combinação das amostras dos estudantes e recenseadores para anular vieses geográficos ou de faixa etária.
2. **Normalização Espacial:** Redimensionamento de todas as imagens para um grid quadrado padrão de 20x20 pixels.
3. **Centralização por Centro de Massa:** Posicionamento dos dígitos no centro geométrico de uma moldura de 28x28 pixels, gerando uma representação normalizada e suavizada por técnicas de anti-aliasing.

O MNIST rapidamente consolidou-se como o padrão de ouro internacional, atuando como o "combustível" fundamental para a validação das primeiras Redes Neurais Convolucionais (*LeNet-5*). O dataset transformou-se no marco zero da Visão Computacional, servindo como o laboratório prático onde a viabilidade das arquiteturas profundas (*Deep Learning*) foi matematicamente comprovada antes da expansão para problemas tridimensionais complexos.

---

## 💻 Engenharia do Projeto e Estrutura do Repositório

Este repositório apresenta a consolidação prática de um pipeline de engenharia focado no treinamento e na compressão matemática de uma rede convolucional sobre a base histórica do MNIST, visando a portabilidade do modelo para sistemas embarcados em ambientes de IoT e inteligência de borda (*Edge AI*).

### 📁 Direcionamento de Artefatos

Todo o desenvolvimento computacional, códigos-fonte funcionais, binários compactados e o respectivo relatório de métricas foram centralizados estritamente dentro da pasta do projeto selecionado:

👉 **[Acesse o Relatório Técnico e Códigos-Fonte em: projetos/1-classificacao-mnist](./projetos/1-classificacao-mnist)**

```text
projeto-pratico-ia/
├── projetos/
│   └── 1-classificacao-mnist/     # 🎯 Módulo Principal (MNIST Classification & Edge Optimization)
│       ├── train_model.py          # Script de construção, treino e validação da CNN
│       ├── optimize_model.py       # Script de conversão e quantização TFLite
│       ├── run_inference.py        # Script de validação de inferência do modelo otimizado
│       ├── model.h5                # Modelo treinado em alta precisão (Keras/HDF5)
│       ├── model.tflite            # Modelo otimizado para Edge AI (TFLite)
│       ├── exemplo_previsao.png    # Amostra gráfica de predição gerada
│       └── README.md               # Relatório técnico detalhado do candidato
└── README.md                       # Documentação principal do repositório
```

---

## 🧭 Guia de Referência do Repositório (Instruções Originais)

Para fins de auditoria do ambiente padronizado de execução e conformidade com o processo seletivo, o fluxo de engenharia foi executado utilizando o contêiner de desenvolvimento embarcado isolado.

### Requisitos e Preparação do Ambiente (Dev Container / Codespaces)
1. **Ativação via Docker Local:** Abertura da pasta do repositório no VS Code -> Comando rápido `F1` -> Seleção de `Dev Containers: Reopen in Container`.
2. **Processamento em Nuvem (GitHub Codespaces):** Inicialização direta pelo navegador de internet através do menu `<> Code` -> Aba `Codespaces` -> Botão `Create codespace on main`.

### Protocolo de Validação e Entrega
A automação de integração contínua do repositório (`.github/workflows/ci.yml`) executa a checagem com base na higienização de diretórios. O pipeline foi configurado sob as seguintes restrições:
* Eliminação obrigatória das pastas sobressalentes (`2-classificacao-cifar10` e `3-deteccao-mascaras-faciais`) mantendo unicamente a pasta do MNIST.
* Verificação estática dos arquivos binários gerados localmente. A submissão final foi integrada à branch principal via comandos de sincronização do Git.
