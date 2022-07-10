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
        self.h_env = 0
        self.X1 = 0
        self.maxfreqs = 0
        self.amplitudes = 0
        self.phases = 0
        self.sound = 0
        self.env_temp = 0
        self.filter = 0
        self.data_before_filtering = 0

        self.initialize_values(pre_filter)


    def initialize_values(self, pre_filter):
        self.create_window()
        self.create_windowData()
        self.calculate_filter_1000Hz()
        if pre_filter != False:
            self.apply_filtre_1000Hz()
        self.extract_plot_fft()
        self.get_attributs_max()
        self.create_time_envelop()
        wavfile.write(self.file_dir_str + "_after_synth_V2.wav", self.Fs, (self.get_temp_env()*self.synth_signal(0)).astype(np.int16))

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

    def get_data_raw(self):
        return self.data

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
        self.h_env = np.ones(Ordre + 1) * 1 / (Ordre + 1) #L'enveloppe de la moyenne dans le temps
        self.env_temp = np.convolve(self.h_env, np.abs(self.data))[:(len(self.data))] #Covolution fait, car dans le domaine temporel
        return self.env_temp[:(len(self.data))] #Réduire le table pour avoir seulement 160 000 datas

    def find_the_K(self, max_k_value = 2000, freq_cut = np.pi/1000):
        for K in range(1, max_k_value):
            value_pi_1000 = (1/K)*(np.sin(freq_cut*K/2))/np.sin(freq_cut/2)
            if(value_pi_1000 > 0.707) and (value_pi_1000 < 0.708):
                return K
        return 0

    def calculate_filter_1000Hz(self):
        K = 3
        N = 1024
        Filtre0 = np.zeros(1)
        self.filter = np.zeros(N-1)
        Filtre0[0] = 1 - 6 / 1024
        n = np.arange(1, N)
        self.filter = - 2 * ((1/1024) * ((np.sin(np.pi * n * K / N)) / (np.sin(np.pi * n / N))) * np.cos(20 * np.pi * n / 441)) #Filter dans le domaine du temps (Réponse impulsionnelle)
        self.filter = np.concatenate((Filtre0, self.filter)) #Ajout de la valeur 0 et de la formule du filtre dans le temps
        self.data_before_filtering = self.data


    def apply_filtre_1000Hz(self):
        result = np.convolve(self.data_window, self.filter)
        self.data = result[:(len(self.data))]
        wavfile.write("basson_filtre_V2", self.Fs, result.astype(np.int16))
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
        self.X1 = np.fft.fft(self.data) #change for window

    def get_samplerate(self):
        return self.Fs

    def get_sound(self):
        return self.sound

    def get_temp_env(self):
        return self.env_temp

    def show_FFT_result_before_filtering(self):
        X1_before = np.fft.fft(self.data_before_filtering)
        halflen = int(len(X1_before) / 2)
        n = np.arange(0, len(self.data))
        freq = n[:(halflen - 1)] / (len(X1_before) / self.Fs)
        log_fft_result = 20 * np.log10(abs(X1_before[:(halflen - 1)]))
        plt.title("FFT du signal entrant avant le filtrage de " + self.file_dir_str)
        plt.plot(freq, log_fft_result)
        plt.show()

    def show_FFT_result(self):
        halflen = int(len(self.X1) / 2)
        n = np.arange(0, len(self.data))
        freq = n[:(halflen - 1)] / (len(self.X1) / self.Fs)
        log_fft_result = 20 * np.log10(abs(self.X1[:(halflen - 1)]))
        plt.title("FFT du signal entrant de " + self.file_dir_str)
        plt.plot(freq, log_fft_result)
        plt.show()

    def show_env_temp(self):
        plt.plot(self.env_temp)
        plt.title("Affichage de l'enveloppe")
        plt.show()

    def show_filter(self):
        freq = (np.arange(-1024/2, 1024/2)*self.Fs/1024)
        plt.title("FFT du filtre de 1000Hz")
        plt.plot(freq, np.fft.fftshift(np.fft.fft(self.filter)))
        plt.show()

    def show_fft_envelop(self):
        n = np.arange((-884/2), (884/2)+1)
        n_freq = n*self.Fs/884
        h_env_fft = np.fft.fftshift(np.fft.fft(self.h_env))
        plt.plot(n_freq,h_env_fft)
        plt.title("FFT du Filtre pour l'enveloppe")
        #plt.plot(20*np.log10(h_env_fft)) doesn't work
        plt.show()

    def test_filter(self):
        n = np.arange(0, self.data_length)
        sinus_function = np.sin(2 * np.pi * n * 1000 / self.Fs)

        result = np.convolve(sinus_function, self.filter)
        result = result[:(len(sinus_function))]
        plt.title("Sinus de 1000Hz filtrée")
        plt.plot(result)
        plt.show()




def main():
    guitar_synth = analyse_audio_file("note_guitare_LAd.wav", pre_filter=False)
    basson_synth  = analyse_audio_file("note_basson_plus_sinus_1000_Hz.wav", pre_filter=True)

    SOL = basson_synth.get_temp_env() * basson_synth.synth_signal(-3)
    FA = basson_synth.get_temp_env() * basson_synth.synth_signal(-5)
    RE = basson_synth.get_temp_env() * basson_synth.synth_signal(-8)
    MIb = basson_synth.get_temp_env() * basson_synth.synth_signal(-7)

    demi = 22175
    un = 44350
    deux = 88700
    silence = np.zeros(demi)
    start = 21000
    args = (SOL[start:start + demi], SOL[start:start + demi], SOL[start:start + demi],
                             MIb[start:start + deux], silence, FA[start:start + demi], FA[start:start + demi],
                             FA[start:start + demi], RE[start:start + deux])

    musique = np.concatenate(args)
    wavfile.write("musique2.wav", basson_synth.get_samplerate(), musique.astype(np.int16))
    basson_synth.show_FFT_result()
    basson_synth.show_FFT_result_before_filtering()





if __name__ == '__main__':
    main()