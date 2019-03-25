import librosa
import numpy as np
import pysndfile


def load_audio_sec(file_path, sr=16000):
    data, loaded_sr = librosa.core.load(file_path, sr)
    if sr is not None and len(data) > sr:
        data = data[:sr]
    elif sr is not None:
        data = np.pad(data, pad_width=(0, max(0, sr - len(data))), mode="constant")
    return data, loaded_sr


def write_audio_file(file_path, audio, sr=16000):
    if sr is None:
        sr = 48000    
    pysndfile.sndio.write(file_path, audio, rate=sr, format='wav', enc='pcm16')
#    librosa.output.write_wav(file_path, audio, sr)


supported_formats = ['.wav', '.flac']


def supported_format(file):
    return np.any([file.endswith(ext) for ext in supported_formats])
