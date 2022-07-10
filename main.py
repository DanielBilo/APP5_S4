import wave as wv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal as scp




class analyse_audio_file:
    def __init__(self, file_dir_str, windowType = np.hanning, pre_filter = False):
        self.note_dict = 0
        self.file_dir_str = file_dir_str
        self.Fs, self.data = wavfile.read(file_dir_str)
        self.windowType = windowType
        self.data_length = len(self.data)
        self.window = 0
        self.data_window = 0
        self.X1 = 0
        self.maxfreqs = 0
        self.amplitudes = 0
        self.phases = 0
        self.sound = 0
        self.env_temp = 0

        self.initialize_values(pre_filter)


    def initialize_values(self, pre_filter):
        self.create_window()
        self.create_windowData()
        if pre_filter != False:
            self.filtre_1000Hz()
        self.extract_plot_fft()
        self.get_attributs_max()
        self.create_time_envelop()

    def get_attributs_max(self, numberOfValues = 32):
        peak_distance_values = {"note_guitare_LAd.wav":1500, "note_basson_plus_sinus_1000_Hz.wav": 600}
        peak_distance = peak_distance_values[self.file_dir_str]
        maxindex = np.argmax(self.X1)
        index_harm = np.arange(0, numberOfValues)
        for x in range(32):
            index_harm[x] = maxindex * x + maxindex
        find_peak = scp.find_peaks(self.X1[:(80000)], distance=peak_distance)

        self.maxfreqs = np.asarray(find_peak, dtype=object)[0]
        self.amplitudes = np.absolute(self.X1[self.maxfreqs[1:33]])
        self.phases = np.angle(self.X1[self.maxfreqs[1:(33)]])

    def synth_signal(self, note):
        n = np.arange(0, len(self.data))
        self.sound = np.zeros(len(self.data))
        for x in range(32):
            self.sound = self.sound + self.amplitudes[x] * np.sin(2 * np.pi * ((self.maxfreqs[x + 1] / (len(self.X1)) * self.Fs) * (2 ** (note / 12))) * n / self.Fs + self.phases[x])
        self.sound = self.sound/100000000
        return self.sound

    def create_time_envelop(self):
        h0 = np.zeros(1)
        h0[0] = 1
        Ordre = self.find_the_K() - 1  # K = Ordre + 1
        h_env = np.ones(Ordre + 1) * 1 / (Ordre + 1) #L'enveloppe de la moyenne dans le temps
        self.env_temp = np.convolve(h_env, np.abs(self.data_window))[:(len(self.data))] #Covolution fait, car dans le domaine temporel
        return self.env_temp[:(len(self.data))] #Réduire le table pour avoir seulement 160 000 datas

    def find_the_K(self, max_k_value = 2000, freq_cut = np.pi/1000):
        for K in range(1, max_k_value):
            value_pi_1000 = (1/K)*(np.sin(freq_cut*K/2))/np.sin(freq_cut/2)
            if(value_pi_1000 > 0.707) and (value_pi_1000 < 0.708):
                return K
        return 0

    def filtre_1000Hz(self):
        K = 3
        N = 1024
        Filtre0 = np.zeros(1)
        Filtre0[0] = 1 - 6 / 1024
        n = np.arange(1, N)
        Filtre = - 2 * ((1/1024) * ((np.sin(np.pi * n * K / N)) / (np.sin(np.pi * n / N))) * np.cos(20 * np.pi * n / 441)) #Filter dans le domaine du temps (Réponse impulsionnelle)
        Filtre = np.concatenate((Filtre0, Filtre)) #Ajout de la valeur 0 et de la formule du filtre dans le temps
        result = np.convolve(self.data_window, Filtre)
        self.data = result[:(len(self.data))]
        return result

    def create_synth_note(self, index_harmonique):
        return self.get_temp_env() * self.synth_signal(index_harmonique)

    def create_window(self):
        self.window = self.windowType(self.data_length)

    def change_windowTpe(self, windowType):
        self.windowType = windowType

    def create_windowData(self):
        self.data_window = self.window*self.data

    def extract_plot_fft(self):
        self.X1 = np.fft.fft(self.data_window) #change for window

    def get_samplerate(self):
        return self.Fs

    def get_sound(self):
        return self.sound

    def get_temp_env(self):
        return self.env_temp



    def show_FFT_result(self):
        halflen = int(len(self.X1) / 2)
        n = np.arange(0, len(self.data))
        freq = n[:(halflen - 1)] / (len(self.X1) / self.Fs)
        log_fft_result = 20 * np.log10(abs(self.X1[:(halflen - 1)]))
        plt.plot(freq, log_fft_result)
        plt.show()


def main():
    guitar_synth = analyse_audio_file("note_guitare_LAd.wav", pre_filter=False)
    basson_synth  = analyse_audio_file("note_basson_plus_sinus_1000_Hz.wav", pre_filter=True)

    SOL = guitar_synth.get_temp_env() * guitar_synth.synth_signal(-3)
    FA = guitar_synth.get_temp_env() * guitar_synth.synth_signal(-5)
    RE = guitar_synth.get_temp_env() * guitar_synth.synth_signal(-8)
    MIb = guitar_synth.get_temp_env() * guitar_synth.synth_signal(-7)

    demi = 22175
    un = 44350
    deux = 88700
    silence = np.zeros(demi)
    start = 10000
    args = (SOL[start:start + demi], SOL[start:start + demi], SOL[start:start + demi],
                             MIb[start:start + deux], silence, FA[start:start + demi], FA[start:start + demi],
                             FA[start:start + demi], RE[start:start + deux])

    musique = np.concatenate(args)
    wavfile.write("musique2.wav", guitar_synth.get_samplerate(), musique.astype(np.int16))





if __name__ == '__main__':
    main()