import os
import tempfile

import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import laughter_classification.psf_features as psf_features


class FeatureExtractor:
    def extract_features(self, wav_path):
        """
        Extracts features for classification ny frames for .wav file

        :param wav_path: string, path to .wav file
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """
        raise NotImplementedError("Should have implemented this")


class PyAAExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""

    def __init__(self):
        self.extract_script = "./extract_pyAA_features.py"
        self.py_env_name = "ipykernel_py2"

    def extract_features(self, wav_path):
        with tempfile.NamedTemporaryFile() as tmp_file:
            feature_save_path = tmp_file.name
            cmd = "python \"{}\" --wav_path=\"{}\" " \
                  "--feature_save_path=\"{}\"".format(self.extract_script, wav_path, feature_save_path)
            os.system("source activate {}; {}".format(self.py_env_name, cmd))

            feature_df = pd.read_csv(feature_save_path)
        return feature_df


class Extractor(FeatureExtractor):

    def __init__(self, frame_length, frame_step=20):
        self.frame_length = frame_length
        self.step = frame_step

    # extracts MFCC and Mel spectrogram features
    def extract_features(self, wav_path):
        rate, data = wavfile.read(wav_path)
        data_len = data.shape[0]
        frame = int(rate * self.frame_length)
        return np.array(
            [np.mean(librosa.feature.mfcc(data[start: start + frame], rate).T, axis=0)
             for start in range(0, data_len - frame + 1, self.step)]
        ), np.array(
            [np.mean(librosa.feature.melspectrogram(data[start: start + frame], rate), axis=0)
             for start in range(0, data_len - frame + 1, self.step)]
        )
