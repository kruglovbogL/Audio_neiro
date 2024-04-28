from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.io import wavfile
from scipy.signal import welch
from scipy.fft import fft
import  librosa
import matplotlib.pyplot as plt
import librosa.display as ld
import sounddevice as sd
import json
import librosa.feature
import pandas as pd

signal,sr = librosa.load('barrel.wav')   #load file



n_fft = 1000
#D = np.abs(librosa.stft(signal))
D = np.abs(librosa.stft(signal,n_fft=n_fft, hop_length=n_fft//2,win_length=n_fft))**2


#Получить спектр мощности
#power_spectrum = librosa.amplitude_to_db(D,ref=np.max)
power_spectrum = np.mean(D,axis=1)

#Вычисление частоты
#frequencies = librosa.fft_frequencies(sr=sr)
frequencies = np.arange(0,n_fft/2+1)*(sr/n_fft)

# np.mean(power_spectrum,axis=1)

spectrum_in_db = librosa.power_to_db(power_spectrum,ref=np.max)
#visual

plt.figure(figsize=(12,6))
# ld.waveshow(signal, sr=sr)

#np.mean(power_spectrum,axis=1)

plt.plot(frequencies, spectrum_in_db)
plt.xlabel('Частота (Гц)')
plt.ylabel('Мощность (dB)')
plt.title('Power Spectrum')
plt.xscale('log')
plt.xlim(0, sr/2)
# plt.ylim([0, -30])
plt.grid(True)
plt.show()

rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
roloff_mean = np.mean(rolloff)
croloff_std = np.std(rolloff)

mfcc_mean = np.mean(librosa.feature.mfcc(y=signal, sr=sr), axis=1)
mfcc_std = np.std(librosa.feature.mfcc(y=signal, sr=sr), axis=1)
cent = librosa.feature.spectral_centroid(y=signal, sr=sr)

cent_mean = np.mean(cent)
cent_std = np.std(cent)
zrate=librosa.feature.zero_crossing_rate(signal)

nme = ["aparna", "pankaj", "sudhir", "Geeku"]
deg = ["MBA", "BCA", "M.Tech", "MBA"]
scr = [90, 40, 80, 98]

# dictionary of lists
# mfcc_mean, mfcc_std,cent_mean,cent_std,roloff_mean, croloff_std,zrate
dict = {'signal' : signal}

df = pd.DataFrame(dict)


df.to_csv('scv_data_signal.csv')
print(pd.read_csv('scv_data_signal.csv'))
df['signal'].unique()