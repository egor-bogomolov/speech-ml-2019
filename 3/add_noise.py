from argparse import ArgumentParser
import os
from NoiseLoader import NoiseLoader
from util import load_audio_sec, write_audio_file, supported_format

noises_dirs = ['bg_noise/FRESOUND_BEEPS_gsm/train', 'bg_noise/freesound_background_gsm']


def process_file(path, file, output_directory, noise_loader, sr, nr):
    audio, loaded_sr = load_audio_sec(os.path.join(path, file), sr)
    noised_audio = add_noise_to_audio(audio, noise_loader, nr)
    write_audio_file(os.path.join(output_directory, file), noised_audio, loaded_sr)


def add_noise_to_audio(audio, noise_loader, nr):
    noised_audio = audio.copy()
    noise = noise_loader.random_noise()
    for i in range(0, len(audio), len(noise)):
        size = min(len(noise), len(audio) - i)
        noised_audio[i:i + size] += noise[:size] * nr
    return noised_audio


def process_directory(directory, output_directory, noise_loader, sr, nr):
    for path, dirs, files in os.walk(directory):
        for file in files:
            if supported_format(file):
                process_file(path, file, output_directory, noise_loader, sr, nr)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--directory', help='Directory containing files for augmentation.', required=True)
    parser.add_argument('-o', '--output', help='Directory for noised files.', default='noised_samples/')
    parser.add_argument('-sr', '--sample_rate', help='Sample rate for audio files.', default=16000, type=int)
    parser.add_argument('-nr', '--noise_rate', help='Noise / sound ratio.', default=0.05, type=float)
    args = parser.parse_args()

    process_directory(args.directory, args.output, NoiseLoader(noises_dirs), args.sample_rate, args.noise_rate)
