import wave as wv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

samplerate, data = wavfile.read('note_guitare_LAd.wav')
#FFT signal LAd
window = np.hamming(len(data))
dataw = window * data
X1 = np.fft.fft(dataw)
halflen = int(len(X1)/2)
n = np.arange(0, len(data))
Fs = 44100
freq = n[:(halflen - 1)] / (len(X1) / Fs)

plt.plot(freq, np.log10(abs(X1[:(halflen - 1)])))
plt.show()

maxfreqs = np.argpartition(X1, -32)[-32:]
amplitudes = np.absolute(X1[maxfreqs])
phases = np.angle(X1[maxfreqs])

# début FIR
N = 1024
# ((fc_rad_ech/np.pi)*N) + 1
K = 3
w1 = np.pi / 1000
h0 = np.zeros(1)
h0[0] = K/N

n = np.arange(1, N)
RepImp = np.concatenate((h0, (np.sin(np.pi * n * K / N))/(N * np.sin(np.pi * n/N))))
# print(RepImp)
absdata = abs(data)
result = np.convolve(absdata, abs(RepImp))

# plt.plot(resultw)
plt.plot(abs(result))
plt.show()

