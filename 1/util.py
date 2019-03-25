import librosa
import numpy as np


def load_audio_sec(file_path, sr=16000):
    data, _ = librosa.core.load(file_path, sr)
    if len(data) > sr:
        data = data[:sr]
    else:
        data = np.pad(data, pad_width=(0, max(0, sr - len(data))), mode="constant")
    return data


def write_audio_file(file_path, audio, sr=16000):
    librosa.output.write_wav(file_path, audio, sr)


supported_formats = ['.wav', '.flac']


def supported_format(file):
    return np.any([file.endswith(ext) for ext in supported_formats])


