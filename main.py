import wave as wv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal as scp
import xlsxwriter



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
        self.pure_data = 0
        self.peak = 0

        self.initialize_values(pre_filter)


    def initialize_values(self, pre_filter):
        self.pure_data = self.data
        self.calculate_filter_1000Hz()
        filtered = False
        if pre_filter != False:
            self.apply_filtre_1000Hz()
            filtered = True
        self.create_window()
        self.create_windowData()
        self.create_time_envelop()
        self.extract_plot_fft(filtered)
        self.get_attributs_max()

        wavfile.write(self.file_dir_str + "_after_synth_V2.wav", self.Fs, (self.get_temp_env()*self.synth_signal(0)).astype(np.int16))

    def get_attributs_max(self, numberOfValues = 32):
        peak_distance_values = {"note_guitare_LAd.wav":1500, "note_basson_plus_sinus_1000_Hz.wav": 500}
        peak_distance = peak_distance_values[self.file_dir_str]
        maxindex = np.argmax(self.X1)
        find_peak = scp.find_peaks(self.X1[:(80000)], distance=peak_distance)

        self.maxfreqs = np.asarray(find_peak, dtype=object)[0]
        self.amplitudes = np.absolute(self.X1[self.maxfreqs[1:33]])
        self.phases = np.angle(self.X1[self.maxfreqs[1:(33)]])

        workbook = xlsxwriter.Workbook(self.file_dir_str.rstrip(".wave") + '.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write(0,0,"frequency")
        worksheet.write(0,1,"amplitude")
        worksheet.write(0, 2, "angle")

        for row in range(1,33):
            # write operation perform
            worksheet.write(row, 0, (self.maxfreqs[row -1]/self.data_length)*self.Fs)
            worksheet.write(row, 1, self.amplitudes[row-1])
            worksheet.write(row, 2, self.phases[row-1])
            row += 1
        workbook.close()


    def get_data_raw(self):
        return self.data

    def synth_signal(self, note):
        n = np.arange(0, len(self.data))
        self.sound = np.zeros(len(self.data))
        for x in range(32):
            self.sound = self.sound + self.amplitudes[x] * np.sin(2 * np.pi * ((self.maxfreqs[x + 1] / (len(self.X1)) * self.Fs) * (2 ** (note / 12))) * n / self.Fs + self.phases[x])
        if self.file_dir_str.rsplit("_")[1] == "basson":
            self.sound = self.sound/10000
        else:
            self.sound = self.sound / 1600
        return self.sound

    def create_time_envelop(self):
        h0 = np.zeros(1)
        h0[0] = 1
        Ordre = self.find_the_K() - 1  # K = Ordre + 1
        self.h_env = np.ones(Ordre + 1) * 1 / (Ordre + 1) #L'enveloppe de la moyenne dans le temps
        self.env_temp = np.convolve(self.h_env, np.abs(self.data))[:(len(self.data))] #Covolution fait, car dans le domaine temporel
        self.peak = np.ptp(self.env_temp)
        self.env_temp = self.env_temp /self.peak
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
        result = np.convolve(self.data, self.filter)
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

    def extract_plot_fft(self, filtered):
        if filtered:
            self.X1 = np.fft.fft(self.data_window)
        else:
            self.X1 = np.fft.fft(self.data_window)

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

        plt.xlabel('Fréquences')
        plt.ylabel('Amplitude (dB)')
        plt.title("FFT du signal entrant de " + self.file_dir_str)
        plt.plot(freq, log_fft_result)
        plt.show()

    def show_cutband_impulsion(self):
        plt.xlabel('échantillon')
        plt.ylabel('Amplitude')
        plt.title("Valeur de h(n) du filtre coupe-bande " + self.file_dir_str)
        plt.plot(self.filter)
        plt.show()

    def show_env_temp(self):
        plt.xlabel('Échantillon')
        plt.ylabel('Amplitude')
        plt.plot(self.data)
        plt.plot(self.env_temp*self.peak)
        plt.title("Données avec la fenêtre appliqué sur le fichier audio du/de la " + self.file_dir_str.rsplit("_")[1])
        plt.legend(["Données du fichier", "Valeurs de l'enveloppe temporelle"])
        plt.show()

    def show_filter(self):
        freq = (np.arange(-1024/2, 1024/2)*self.Fs/1024)
        plt.title("FFT du filtre coupe-bande de 1000Hz")
        plt.xlabel('fréquence')
        plt.ylabel('Amplitude (dB)')
        plt.xlim(0,2000)
        plt.stem(freq, 20*np.log10(np.fft.fftshift(np.fft.fft(self.filter))))
        plt.show()

    def show_Frequency_Response_envelop(self, xlimit = np.pi):
        K = 885
        n = np.arange(-np.pi, np.pi, 0.00001)
        X = 20*np.log10(np.abs((1 / K) * (np.sin(n * K / 2)) / np.sin(n / 2)))
        plt.xlim(-xlimit, xlimit)
        plt.plot(n,X)
        plt.xlabel('Radian/échantillon')
        plt.ylabel('Amplitude (dB)')
        plt.title("Réponse en fréquence (TFSD) du filtre de l'enveloppe temporel")
        plt.show()

    def test_filter(self):
        n = np.arange(0, self.data_length)
        sinus_function = np.sin(2 * np.pi * n * 1000 / self.Fs)

        result = np.convolve(sinus_function, self.filter)
        result = result[:(len(sinus_function))]
        plt.xlabel('échantillon')
        plt.ylabel('Amplitude')
        plt.title("Sinus de 1000Hz filtrée")
        plt.plot(result)
        plt.show()

    def create_symph_mega_cool(self):
        SOL = self.create_synth_note(-3)
        FA = self.create_synth_note(-5)
        RE = self.create_synth_note(-8)
        MIb = self.create_synth_note(-7)

        demi = 22175
        un = 44350
        deux = 88700
        silence = np.zeros(demi)
        start = 21000
        args = (SOL[start:start + demi], SOL[start:start + demi], SOL[start:start + demi],
                MIb[start:start + deux], silence, FA[start:start + demi], FA[start:start + demi],
                FA[start:start + demi], RE[start:start + deux])
        musique = np.concatenate(args)
        wavfile.write(self.file_dir_str.rsplit("_")[1] + "_musique_mega_cool.wav", self.get_samplerate(), musique.astype(np.int16))

    def show_amplitude_phase_cutband(self, x_limit = 1024/2):

        X_filter = np.fft.fft(self.filter)[0:int(len(self.filter)/2)]
        fig, axs = plt.subplots(2)
        fig.suptitle('amplitude et phase de la réponse en fréquence du filtre coupe-bande')
        axs[0].plot(np.absolute(X_filter))
        axs[0].set_xlim(0,x_limit)
        axs[0].set_ylabel("Amplitude")
        axs[1].plot(np.angle(X_filter))
        axs[1].set_ylabel("phase")
        axs[1].set_xlabel("échantillon")
        axs[1].set_xlim(0,x_limit)
        plt.show()

    def show_FFT_signal_synth(self):

        X_synth = np.fft.fft(self.create_synth_note(0))
        halflen = int(len(X_synth) / 2)
        n = np.arange(0, len(self.data))
        freq = n[:(halflen - 1)] / (len(X_synth) / self.Fs)
        log_fft_result = 20 * np.log10(abs(X_synth[:(halflen - 1)]))

        plt.xlabel('Fréquences')
        plt.ylabel('Amplitude (dB)')
        plt.title("FFT du signal synthonisé de " + self.file_dir_str)
        plt.plot(freq, log_fft_result)
        plt.show()





def main():
    guitar_synth = analyse_audio_file("note_guitare_LAd.wav", windowType = np.blackman, pre_filter=False)
    basson_synth  = analyse_audio_file("note_basson_plus_sinus_1000_Hz.wav",windowType = np.blackman, pre_filter=True)


    guitar_synth.show_FFT_result()
    guitar_synth.show_FFT_signal_synth()
    basson_synth.show_FFT_result()
    basson_synth.show_FFT_signal_synth()
    guitar_synth.show_env_temp()
    basson_synth.show_env_temp()
    guitar_synth.create_symph_mega_cool()
    basson_synth.create_symph_mega_cool()
    # basson_synth.show_Frequency_Response_envelop(2*np.pi/1000)





if __name__ == '__main__':
    main()