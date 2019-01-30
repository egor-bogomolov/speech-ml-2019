## Audio data augmentation

### How to run the script

#### Dependencies

* [numpy](https://www.numpy.org/)
* [librosa](https://librosa.github.io/librosa/)

#### Run the script

To process all audio files in directory `samples` and save noised versions in `noised_samples` run:

~~~
python add_noise.py -d samples/ -o noised_samples/
~~~

Also you can set sample rate and noise / sound ratio:

~~~
python add_noise.py -d samples/ -nr 0.1 -sr 32000
~~~

#### Restrictions

* Supported formats: .wav, .flac