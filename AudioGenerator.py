import numpy as np
import matplotlib.pyplot as plt
import wave
import struct

class AudioGenerator:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
    
    def generate_sine_wave(self, frequency, duration, amplitude):
        num_samples = int(self.sample_rate * duration)
        samples = np.arange(num_samples)
        waveform = amplitude * np.sin(2 * np.pi * frequency * samples / self.sample_rate)
        return waveform.astype(np.int16)
        #https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html
        #Section 1 Sine Waves
        #Section 2 Sampling
        #Section 4 16 Bit 
        #cast to type int 16 to keep values within 16bit range
    
    def generate_square_wave(self, frequency, duration, amplitude):
        num_samples = int(self.sample_rate * duration)
        samples = np.arange(num_samples)
        w = 2 * np.pi * frequency 
        sq_wave = amplitude * (4/np.pi) * ((np.sin(w*(samples / self.sample_rate))) + 1/3*np.sin(3*w*(samples / self.sample_rate)) + 1/5*np.sin(5*w*(samples / self.sample_rate)) + 1/7*np.sin(7*w*(samples / self.sample_rate)) + 1/9*np.sin(9*w*(samples / self.sample_rate)) + 1/11*np.sin(11*w*(samples / self.sample_rate)) + 1/13*np.sin(13*w*(samples / self.sample_rate)))
        return sq_wave.astype(np.int16)
    
    def add_noise_to_wave(self, waveform, noise_amplitude):
        noise = np.random.randint(-noise_amplitude, noise_amplitude, len(waveform))
        noisy_waveform = np.clip(waveform + noise, -32768, 32767)
        return noisy_waveform.astype(np.int16)
    #Section 3 Noise 

    def save_to_wav(self, filename, waveform):
    #https://docs.python.org/3/library/wave.html
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.setcomptype('NONE', 'not compressed')

            for sample in waveform:
                packed_sample = struct.pack('h', sample) 
                #h short integer c type
                wav_file.writeframes(packed_sample)

    def read_wav_file(self, filename):
    #https://docs.python.org/3/library/wave.html
        with wave.open(filename, 'r') as wav_file:
            num_frames = wav_file.getnframes()
            wav_data = wav_file.readframes(num_frames)
            waveform = np.frombuffer(wav_data, dtype=np.int16)
        return waveform

    def plot_waveform(self, waveform):
    #https://matplotlib.org/stable/index.html
        duration = len(waveform) / self.sample_rate
        time = np.linspace(0, duration, len(waveform))
        plt.figure()
        plt.plot(time, waveform)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Waveform')
        plt.show()
        
audio = AudioGenerator(sample_rate=44100)
#EX:
square_wave = audio.generate_square_wave(349, 2, 10000)
square_wave2 = audio.generate_square_wave(440, 2, 10000)
square_wave3 = audio.generate_square_wave(261, 2, 10000)

final_square = np.array([], dtype=np.int16) 

final_square= np.append(final_square, square_wave)
final_square= np.append(final_square, square_wave2)
final_square= np.append(final_square, square_wave3)

audio.save_to_wav("final_square.wav",final_square)
