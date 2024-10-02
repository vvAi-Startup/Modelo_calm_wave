import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas
from tensorflow.keras.models import load_model
from pydub import AudioSegment
from pydub.playback import play

# Função para carregar e gerar Mel-Spectrograma a partir de um arquivo de áudio
def carregar_audio_gerar_melspec(caminho, sr=22050, n_mels=128):
    y, sr = librosa.load(caminho, sr=sr)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    # Redimensionar para formato esperado
    if melspec_db.shape[1] >= 128:
        melspec_db = melspec_db[:, :128]
    melspec_db = melspec_db[np.newaxis, np.newaxis, ..., np.newaxis]  # Ajustar para a entrada da CNN

    return melspec_db

# Função para exibir o espectrograma
def exibir_spectrograma(caminho, canvas):
    y, sr = librosa.load(caminho, sr=22050)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    plt.figure(figsize=(5, 2))
    librosa.display.specshow(melspec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrograma')
    plt.tight_layout()

    # Salvar o gráfico como imagem
    plt.savefig("spectrogram.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    # Carregar a imagem no canvas
    img = tk.PhotoImage(file="spectrogram.png")
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.image = img  # Manter uma referência à imagem

# Função para carregar o arquivo e fazer a previsão
def realizar_previsao(caminho_audio, modelo, rotulos, label_resultado, canvas):
    melspec_db = carregar_audio_gerar_melspec(caminho_audio)

    # Fazer a previsão
    predicao = modelo.predict(melspec_db)
    classe_prevista = np.argmax(predicao, axis=1)[0]

    label_resultado.config(text=f"Classe prevista: {rotulos[classe_prevista]}")

    # Exibir o espectrograma do áudio
    exibir_spectrograma(caminho_audio, canvas)

# Função para abrir explorador de arquivos
def selecionar_arquivo(label_arquivo, label_resultado, modelo, rotulos, canvas):
    caminho_arquivo = filedialog.askopenfilename(title="Selecione um arquivo de áudio", filetypes=[("Arquivos WAV", "*.wav")])
    if caminho_arquivo:
        label_arquivo.config(text=f"Arquivo selecionado: {os.path.basename(caminho_arquivo)}")
        realizar_previsao(caminho_arquivo, modelo, rotulos, label_resultado, canvas)

def tocar_audio(caminho_audio):
    # Usar pydub para tocar o arquivo de áudio
    audio = AudioSegment.from_wav(caminho_audio)
    play(audio)

# Função para iniciar o player de áudio
def tocar_audio_selecionado(label_arquivo):
    caminho_audio = label_arquivo.cget("text").replace("Arquivo selecionado: ", "")
    if os.path.exists(caminho_audio):
        tocar_audio(caminho_audio)
    else:
        print("Nenhum arquivo de áudio válido selecionado para reprodução.")

# Função principal para a interface
def criar_interface(modelo, rotulos):
    # Janela principal
    root = tk.Tk()
    root.title("Classificação de Áudio")

    # Label para mostrar o arquivo selecionado
    label_arquivo = Label(root, text="Nenhum arquivo selecionado", width=50)
    label_arquivo.pack(pady=10)

    # Botão para selecionar o arquivo
    btn_selecionar = Button(root, text="Selecionar arquivo de áudio", command=lambda: selecionar_arquivo(label_arquivo, label_resultado, modelo, rotulos, canvas))
    btn_selecionar.pack(pady=10)

    # Label para mostrar o resultado da previsão
    label_resultado = Label(root, text="", width=50)
    label_resultado.pack(pady=10)

    # Botão para tocar o áudio
    btn_tocar_audio = Button(root, text="Tocar áudio", command=lambda: tocar_audio_selecionado(label_arquivo))
    btn_tocar_audio.pack(pady=10)

    # Canvas para exibir o espectrograma
    canvas = Canvas(root, width=500, height=200)
    canvas.pack(pady=10)

    # Botão para fechar a janela
    btn_sair = Button(root, text="Sair", command=root.quit)
    btn_sair.pack(pady=10)

    # Iniciar a interface gráfica
    root.mainloop()

if __name__ == '__main__':
    # Definir os rótulos do dataset
    rotulos = ['ambulance', 'dog', 'firetruck', 'traffic']

    # Carregar o modelo treinado
    modelo = load_model('modelo_mel_rnn.h5')

    # Criar a interface gráfica
    criar_interface(modelo, rotulos)
