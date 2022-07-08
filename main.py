import wave as wv
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scp
from scipy.io import wavfile
from scipy.io.wavfile import write

def round_odd(number):
    result = np.round(number)
    if(result % 2):
        return result
    else:
        return result + 1

class analyse_audio_file:
    def __init__(self, file_dir_str, windowType = np.hanning):
        self.samplerate, self.data = wavfile.read(file_dir_str)
        self.windowType = windowType
        self.window = windowType(len(self.data))
        self.data_window = self.data #* p.hamming(len(self.data))
        self.X1 = 0
        self.Fs = 0
        self.maxfreqs = 0
        self.amplitudes = 0
        self.phases = 0
        self.sound = 0


    def extract_plot_fft(self):
        self.X1 = np.fft.fft(self.data_window) #change for window
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
        print('MaxIndex ', type(index_harm))

        self.maxfreqs = index_harm / len(self.X1) * self.Fs
        self.maxfreqs, _ = np.asarray(scp.find_peaks(self.X1[:(80000)], distance=1500))
        print('Maxfreqs = ', self.maxfreqs[1:(33)])
        self.amplitudes = np.absolute(self.X1[self.maxfreqs[1:33]])
        self.phases = np.angle(self.X1[self.maxfreqs[1:(33)]])

        return (self.maxfreqs, self.amplitudes, self.phases)

    def synth_signal(self):
        n = np.arange(0, len(self.data_window))
        self.sound = np.zeros(len(self.data_window))
        for x in range(32):
            self.sound = self.sound + self.amplitudes[x] * np.sin(2 * np.pi * ((self.maxfreqs[x+1] / (len(self.X1)) * self.Fs) * (2 ** (-2/12))) * n / self.Fs + self.phases[x])

        self.sound = self.sound/100000000
        # print(self.sound)
        # plt.plot(self.sound)
        # plt.show()
        return self.sound

    def laboratoire1_Question2(self):
        N = 2048
        fs = 44100

        #Déterminer la fréquence de coupure en rad/échantillon
        fc_rad_ech = np.pi/1000
        # print(fc_rad_ech)

        #Il faut trouver la valeur de K qui permet d'avoir la même valeur que fc_rad_ech
        K = round_odd(fc_rad_ech / np.pi * N + 1)
        K = 7


        print('K = ', K)
        #Générer la réponse impulsionnelle du filtre
        n = np.arange(-N/2, N/2 )
        omega = 2 * np.pi * n / N #ne devrait pas s'appliquer, car h(n) est dans le domaine du temps???
        f_normalise = n / N * 44100

        n_without = np.arange(1, N)
        h = [K/N]
        h0 = np.zeros(1)
        h0[0] = K / N
        time_response = np.sin(np.pi * n_without * K / N) / (N * np.sin(np.pi * n_without / N))
        RepImp = np.concatenate((h0, time_response))

        # plt.subplot(2,1,1)
        # plt.xlabel("time index")
        # plt.plot(n, (np.fft.fftshift(RepImp)))
        # plt.subplot(2,1,2)
        # plt.xlabel("frequency(Hz)")


        # plt.plot(np.log(np.fft.fftshift(np.fft.fft(np.concatenate((RepImp, np.zeros(1)))))), 'ro')
        # plt.xlim([0,100])
        # # plt.ylim([-6, 0])
        # plt.plot(f_normalise, np.log(np.fft.fftshift(np.fft.fft(RepImp))))
        #
        # plt.show()
        #

        # print(h)
        #Il faut ensuite faire une convolution avec le signal en entrée et le filtre FIR. Cela va donner le signal dans le temps
        n = np.arange(0, 129)
        x = abs(self.data_window) #change for data window

        result = np.convolve(x, RepImp)
        # plt.plot(result)
        # plt.show()
        return result[:(len(self.data_window))]

    def get_samplerate(self):
        return self.samplerate

    def get_sound(self):
        return self.sound

def main():
    test = analyse_audio_file("note_guitare_LAd.wav")
    test.extract_plot_fft()
    # test.get_attributs_max()
    test.get_attributs_max()
    # plt.subplot(2, 1, 2)
    # plt.plot(test.laboratoire1_Question2() * test.synth_signal())
    # plt.show()


    temp = test.laboratoire1_Question2() * test.synth_signal()
    # print(test.get_samplerate())
     # wavfile.write("example.wav", test.get_samplerate(), test.get_sound().astype(np.int16) * 500)
    wavfile.write("example.wav", test.get_samplerate(), (temp/2).astype(np.int16))
    # wavfile.write("example.wav", test.get_samplerate(), (test.laboratoire1_Question2() * test.synth_signal() * 500).astype(np.int16))




if __name__ == '__main__':
    main()