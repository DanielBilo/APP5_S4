import wave as wv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal as scp
from scipy.io.wavfile import write



class analyse_audio_file:
    def __init__(self, file_dir_str, windowType = np.hanning):
        self.samplerate, self.data = wavfile.read(file_dir_str)
        # self.data = self.data * np.hanning(len(self.data))
        print(len(self.data))
        self.windowType = windowType
        self.window = windowType(len(self.data))
        self.data_window = self.window * self.data
        # plt.subplot(2,1,1)
        # plt.plot(self.data)
        # plt.show()
        self.X1 = 0
        self.Fs = 0
        self.maxfreqs = 0
        self.amplitudes = 0
        self.phases = 0
        self.sound = 0


    def extract_plot_fft(self):
        self.X1 = np.fft.fft(self.data) #change for window
        halflen = int(len(self.X1) / 2)
        n = np.arange(0, len(self.data))
        self.Fs = self.samplerate
        freq = n[:(halflen - 1)] / (len(self.X1) / self.Fs)

        log_fft_result = 20 * np.log10(abs(self.X1[:(halflen - 1)]))
        # plt.plot(freq, log_fft_result)
        # plt.show()
        # print(np.argmax(self.X1[:(halflen - 1)]) / (len(self.X1) / self.Fs))
        return (freq, log_fft_result)

    def get_attributs_max(self, numberOfValues = 32):
        maxindex = np.argmax(self.X1)
        index_harm = np.arange(0, numberOfValues)

        for x in range(32):
            index_harm[x] = maxindex * x + maxindex

        # self.maxfreqs, _ = np.asarray(scp.find_peaks(self.X1[:(80000)], distance=1500)) #pour guitare
        self.maxfreqs, _ = np.asarray(scp.find_peaks(self.X1[:(80000)], distance=600))  # pour Basson
        # print('Max Freq = ', self.maxfreqs / len(self.X1) * self.Fs)
        self.amplitudes = np.absolute(self.X1[self.maxfreqs[1:33]])
        self.phases = np.angle(self.X1[self.maxfreqs[1:(33)]])

        return (self.maxfreqs, self.amplitudes, self.phases)

    def synth_signal(self, note):
        n = np.arange(0, len(self.data_window))
        self.sound = np.zeros(len(self.data_window))
        for x in range(32):
            self.sound = self.sound + self.amplitudes[x] * np.sin(2 * np.pi * ((self.maxfreqs[x + 1] / (len(self.X1)) * self.Fs) * (2 ** (note / 12))) * n / self.Fs + self.phases[x])

        self.sound = self.sound/100000000
        # print(self.sound)
        # plt.plot(self.sound)
        # plt.show()
        return self.sound

    def laboratoire1_Question2(self):
        N = 884
        fs = 44100
        K = 3
        #Générer la réponse impulsionnelle du filtre
        n = np.arange(0, N)
        omega = 2 * np.pi * n / N
        f_normalise = n/N
        n_without = np.arange(1, N)
        h0 = np.zeros(1)
        h0[0] = 1

        formula_array = np.sin(np.pi * n_without * K/N) / (N * np.sin(np.pi * n_without/ N))
        Ordre = 884  # K = Ordre + 1
        h_env = np.ones(Ordre + 1) * 1 / (Ordre + 1)
        amp_env = np.convolve(h_env, np.abs(self.data))
        print('amp_env = ', len(np.abs(self.data)))
        # plt.plot(amp_env)
        # plt.show()
        # formula_array = (1/K) * (np.sin(w * K / 2) / np.sin(w / 2))
        RepImp = np.concatenate((h0, formula_array))


        #Il faut ensuite faire une convolution avec le signal en entrée et le filtre FIR. Cela va donner le signal dans le temps
        x = abs(self.data)

        X = np.fft.fft(RepImp, len(x) + len(RepImp) - 1)
        H = np.fft.fft(RepImp, len(x) + len(RepImp)-1)
        Y = np.fft.ifft(H * X)
        result = np.convolve(x, RepImp)
        # result = np.fft.ifft(Y)
        # plt.subplot(2,1,1)
        # plt.plot(self.data)
        # plt.subplot(2,1,2)
        # plt.plot(result)
        # plt.show()
        # return result[:(len(self.data))]
        return amp_env[:(len(self.data))]
    def get_samplerate(self):
        return self.samplerate

    def get_sound(self):
        return self.sound

    def filtre_1000Hz(self):
        K = 3
        N = 1024
        Filtre = np.zeros(N-1)
        Filtre0 = np.zeros(1)
        Filtre0[0] = 1 - 6 / 1024
        n = np.arange(1, N)
        Filtre = - 2 * ((1/1024) * ((np.sin(np.pi * n * K / N)) / (np.sin(np.pi * n / N))) * np.cos(20 * np.pi * n / 441))
        Filtre = np.concatenate((Filtre0, Filtre))
        #plotting for testing
        # xaxis = np.arange(-N/2, N/2)
        # n = np.arange(0, N)
        # xaxis = xaxis / N * 44100
        # plt.xlim(-5000, 5000)
        # plt.xlabel('Fréqence(Hz)')
        # plt.plot(xaxis, np.fft.fftshift(np.fft.fft(Filtre)))

        # plt.show()
        result = np.convolve(self.data * np.hanning(len(self.data)), Filtre)
        self.data = result[:(len(self.data))]
        # plt.plot(result)
        # plt.show()
        return result

def conc_func(*args):
    xt=[]
    for a in args:
        xt=np.concatenate(a, axis=1)
    return xt

def main():
    # test = analyse_audio_file("note_guitare_LAd.wav")
    # test.extract_plot_fft()
    # test.get_attributs_max()
    # temp = test.laboratoire1_Question2()/10 * np.concatenate((test.synth_signal(), np.zeros(884)))
    # wavfile.write("example.wav", test.get_samplerate(), (temp).astype(np.int16))

    test = analyse_audio_file("note_guitare_LAd.wav")
    # sounds = test.filtre_1000Hz()
    test.extract_plot_fft()
    test.get_attributs_max()
    SOL = test.laboratoire1_Question2() * test.synth_signal(-3)
    FA = test.laboratoire1_Question2() * test.synth_signal(-5)
    RE = test.laboratoire1_Question2() * test.synth_signal(-8)
    MIb = test.laboratoire1_Question2() * test.synth_signal(-7)

    demi = 22175
    un = 44350
    deux = 88700
    silence = np.zeros(demi)
    start = 21000
    args = (SOL[start:start + demi], SOL[start:start + demi], SOL[start:start + demi],
                             MIb[start:start + deux], silence, FA[start:start + demi], FA[start:start + demi],
                             FA[start:start + demi], RE[start:start + deux])

    musique = np.concatenate(args)
    plt.plot(musique)
    plt.show()

    # wavfile.write("synth_basson.wav", test.get_samplerate(), basson_synth.astype(np.int16))
    # wavfile.write("basson_filtre.wav", test.get_samplerate(), sounds.astype(np.int16))
    wavfile.write("musique.wav", test.get_samplerate() * 2, musique.astype(np.int16))





if __name__ == '__main__':
    main()