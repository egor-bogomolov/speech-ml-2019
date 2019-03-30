import os
import numpy as np
from util import supported_format, load_audio_sec


class NoiseLoader:

    def __init__(self, dirs, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noises = self._load_noises(dirs)
        self.n_noises = len(self.noises)

    def _load_noises(self, noise_dirs):
        noises = []
        for d in noise_dirs:
            for file in os.listdir(d):
                if supported_format(file):
                    noises.append(load_audio_sec(os.path.join(d, file), self.sample_rate)[0])
        return noises

    def random_noise(self):
        return self.noises[np.random.choice(self.n_noises)]
