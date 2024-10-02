# Classificador de Áudio Urbano

Este projeto implementa um classificador de áudio urbano utilizando aprendizado de máquina e processamento de sinais. O sistema é capaz de identificar e classificar diferentes sons urbanos, como ambulâncias, cachorros, caminhões de bombeiros e tráfego.

## Funcionalidades

- Carrega arquivos de áudio WAV
- Gera e exibe espectrogramas Mel dos arquivos de áudio
- Classifica o áudio em uma das quatro categorias: ambulância, cachorro, caminhão de bombeiros ou tráfego
- Reproduz o áudio selecionado
- Interface gráfica amigável para interação do usuário

## Requisitos

Para executar este projeto, você precisará das seguintes bibliotecas Python:

- librosa
- numpy
- matplotlib
- tensorflow
- pydub
- tkinter
- scikit-learn

Você pode instalar as dependências usando o pip:

```
pip install librosa numpy matplotlib tensorflow pydub scikit-learn
```

Nota: O tkinter geralmente já vem instalado com o Python.

## Estrutura do projeto

- `treinamento.py`: Script para treinar o modelo de classificação de áudio.
- `executar.py`: Script principal contendo a lógica do classificador e a interface gráfica.
- `modelo_mel_rnn.h5`: Modelo de rede neural treinado para classificação de áudio.

## Como treinar o modelo

1. Prepare seu conjunto de dados de áudio em um diretório estruturado, onde cada subdiretório representa uma classe.
2. Execute o script `treinamento.py`:

```
python treinamento.py
```

O script irá:
- Carregar os arquivos de áudio e gerar Mel-Espectrogramas
- Criar um modelo CNN + LSTM para classificação
- Treinar o modelo usando os dados preparados
- Salvar o modelo treinado como `modelo_mel_rnn.h5`
- Plotar o histórico de treinamento

## Como usar o classificador

1. Certifique-se de que o modelo treinado `modelo_mel_rnn.h5` está no mesmo diretório que o script principal.
2. Execute o script `executar.py`:

```
python executar.py
```

3. Na interface gráfica:
   - Clique em "Selecionar arquivo de áudio" para escolher um arquivo WAV.
   - O espectrograma do áudio será exibido e a classificação será mostrada.
   - Use o botão "Tocar áudio" para reproduzir o som selecionado.

## Como funciona

### Treinamento:
1. Os arquivos de áudio são carregados e convertidos em Mel-Espectrogramas.
2. Um modelo CNN + LSTM é criado para extrair características e analisar a sequência temporal.
3. O modelo é treinado usando os dados preparados, com técnicas como Early Stopping e Learning Rate Reduction.

### Classificação:
1. O áudio é carregado e convertido em um Mel-Espectrograma.
2. O espectrograma é processado para ter o formato adequado para entrada no modelo.
3. O modelo de aprendizado profundo faz a previsão da classe do áudio.
4. O resultado da classificação é exibido junto com o espectrograma do áudio.

## Limitações

- O modelo atual só classifica quatro tipos de sons urbanos.
- Apenas arquivos WAV são suportados no momento.

## Contribuições

Contribuições para melhorar o projeto são bem-vindas. Por favor, abra uma issue para discutir mudanças propostas ou envie um pull request com suas melhorias.

## Licença

[Insira aqui informações sobre a licença do seu projeto]
