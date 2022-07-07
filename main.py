import wave as wv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.io.wavfile import write



class analyse_audio_file:
    def __init__(self, file_dir_str, windowType = np.hanning):
        self.samplerate, self.data = wavfile.read(file_dir_str)
        print(self.data)
        self.windowType = windowType
        self.window = windowType(len(self.data))
        self.data_window = self.window * self.data
        plt.subplot(2,1,1)
        plt.plot(self.data)
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

        self.maxfreqs = index_harm / len(self.X1) * self.Fs
        self.amplitudes = np.absolute(self.X1[index_harm])
        self.phases = np.angle(self.X1[index_harm])

        return (self.maxfreqs, self.amplitudes, self.phases)

    def synth_signal(self):
        n = np.arange(0, len(self.data_window))
        self.sound = np.zeros(len(self.data_window))
        for x in range(32):
            self.sound = self.sound + self.amplitudes[x] * np.sin(2 * np.pi * self.maxfreqs[x] * n / self.Fs)

        self.sound = self.sound/100000000
        # print(self.sound)
        # plt.plot(self.sound)
        # plt.show()
        return self.sound

    def laboratoire1_Question2(self):
        N = 1024
        fc = 2000
        fs = 44100

        #Déterminer la fréquence de coupure en rad/échantillon
        fc_rad_ech = fc*2*np.pi/fs
        # print(fc_rad_ech)

        #Il faut trouver la valeur de K qui permet d'avoir la même valeur que fc_rad_ech
        # K = ((fc_rad_ech/np.pi)*N) + 1
        K = 3
        # print(K)

        #Générer la réponse impulsionnelle du filtre
        n = np.arange(0, N)
        omega = 2 * np.pi * n / N
        f_normalise = n/N

        n_without = np.arange(1, N)
        h = [K/N]
        h0 = np.zeros(1)
        h0[0] = K/N
        formula_array = np.sin(np.pi*n_without*K/N)/(N * np.sin(np.pi * n_without/ N))
        h = h + formula_array.tolist() #Il faut utiliser la règle de l'Hopital si on a une division par 0
        RepImp = np.concatenate((h0, formula_array))
        # print(RepImp)
        # plt.plot(omega, (RepImp))
        # plt.show()


        # print(h)
        #Il faut ensuite faire une convolution avec le signal en entrée et le filtre FIR. Cela va donner le signal dans le temps
        a1 = 1
        a2 = 0.25
        f1 = 200
        f2 = 3000
        n = np.arange(0, 129)
        x = abs(self.data) #change for data window

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

    plt.subplot(2, 1, 2)
    plt.plot(test.laboratoire1_Question2() * test.synth_signal())
    temp = test.laboratoire1_Question2() * test.synth_signal()
    plt.show()
    # print(test.get_samplerate())
    print(test.get_sound())
    # wavfile.write("example.wav", test.get_samplerate(), test.get_sound().astype(np.int16) * 500)
    wavfile.write("example.wav", test.get_samplerate(), (temp).astype(np.int16))
    # wavfile.write("example.wav", test.get_samplerate(), (test.laboratoire1_Question2() * test.synth_signal() * 500).astype(np.int16))




if __name__ == '__main__':
    main()