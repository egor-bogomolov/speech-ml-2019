import os
import numpy as np
from util import supported_format, load_audio_sec


class NoiseLoader:

    def __init__(self, dirs):
        self.noises = self._load_noises(dirs)
        self.n_noises = len(self.noises)

    def _load_noises(self, noise_dirs):
        noises = []
        for d in noise_dirs:
            for file in os.listdir(d):
                if supported_format(file):
                    noises.append(load_audio_sec(os.path.join(d, file)))
        return noises

    def random_noise(self):
        return self.noises[np.random.choice(self.n_noises)]
