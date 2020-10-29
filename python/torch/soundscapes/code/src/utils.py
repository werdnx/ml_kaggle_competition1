import torch
import torchaudio
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

mixer = torchaudio.transforms.DownmixMono()
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])
SAMPLE_RATE = 16000


# Augmentation
# https://github.com/iver56/audiomentations
def process_file(file_path, freq):
    sound = torchaudio.load(file_path, out=None, normalization=True)
    augmented = augment(samples=sound[0], sample_rate=SAMPLE_RATE)
    # load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
    soundData = mixer(augmented)
    # downsample the audio to ~8kHz
    tempData = torch.zeros([160000, 1])  # tempData accounts for audio clips that are too short
    if soundData.numel() < 160000:
        tempData[:soundData.numel()] = soundData[:]
    else:
        tempData[:] = soundData[:160000]

    soundData = tempData
    soundFormatted = torch.zeros([32000, 1])
    # soundFormatted[:32000] = soundData[::5]  # take every fifth sample of soundData
    soundFormatted[:32000] = soundData[::freq]
    soundFormatted = soundFormatted.permute(1, 0)
    return soundFormatted
