import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, TimeDistributed, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder

# Função para carregar arquivos de áudio e gerar o Mel-Spectrograma
def carregar_audio_gerar_melspec(diretorio, sr=22050, n_mels=128):
    dados = []
    rotulos = []
    
    for root, _, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            if arquivo.endswith('.wav'):
                caminho = os.path.join(root, arquivo)
                rotulo = os.path.basename(root)  # Usamos o nome da pasta como rótulo
                y, sr = librosa.load(caminho, sr=sr)
                
                # Gerar o Mel-Spectrograma
                melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                melspec_db = librosa.power_to_db(melspec, ref=np.max)  # Converter para escala logarítmica
                
                # Redimensionar para uma forma consistente
                if melspec_db.shape[1] >= 128:  # Tamanho mínimo necessário
                    melspec_db = melspec_db[:, :128]
                    dados.append(melspec_db)
                    rotulos.append(rotulo)
    
    return np.array(dados), np.array(rotulos)

# Função para criar o modelo CNN + RNN (LSTM)
def criar_modelo(input_shape, num_classes):
    model = Sequential()
    
    # Camada CNN para extração de características
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(BatchNormalization()))  # Adicionando BatchNormalization
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(BatchNormalization()))  # Aumentando a profundidade da CNN
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    
    model.add(TimeDistributed(Flatten()))
    
    # Camada LSTM para análise temporal
    model.add(LSTM(128, return_sequences=False))
    
    # Camada final de classificação
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Função para plotar os resultados de treinamento
def plotar_historico(historico):
    plt.plot(historico.history['accuracy'], label='Acurácia Treino')
    plt.plot(historico.history['val_accuracy'], label='Acurácia Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.show()

# Função principal para executar o treinamento
def treinar_modelo(diretorio_dados):
    # Carregar dados e gerar Mel-Spectrogramas
    X, y = carregar_audio_gerar_melspec(diretorio_dados)
    
    # Ajustar dimensões dos dados para entrada no modelo CNN + LSTM
    X = X[..., np.newaxis]  # Adicionar canal extra para a CNN
    X = X[:, np.newaxis, ...]  # Ajustar para TimeDistributed

    # Converter rótulos de string para inteiros
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Dividir os dados em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    
    # Definir parâmetros
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    # Criar o modelo
    modelo = criar_modelo(input_shape, num_classes)
    
    # Definir callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('modelo_mel_rnn.h5', monitor='val_accuracy', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    
    # Treinar o modelo
    historico = modelo.fit(X_train, y_train, validation_data=(X_val, y_val),
                           epochs=50, batch_size=32, callbacks=[early_stopping, checkpoint, reduce_lr])
    
    # Plotar os resultados do treinamento
    plotar_historico(historico)

if __name__ == '__main__':
    diretorio_dados = './dataset/'
    treinar_modelo(diretorio_dados)
