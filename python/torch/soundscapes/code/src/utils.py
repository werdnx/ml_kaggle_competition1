import librosa
import torch
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# mixer = torchaudio.transforms.DownmixMono()
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])
SAMPLE_RATE = 16000
DEF_FREQ = 5


# Augmentation
def process_file(file_path, freq=DEF_FREQ, train=True):
    # sound = torchaudio.load(file_path, out=None, normalization=True)
    sound, r = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    sound = librosa.util.normalize(sound, axis=0)
    # load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
    # sound_np = sound[0].numpy()
    # sound_np = sound
    # if sound_np.shape[0] == 2:
    #     sound_data = librosa.to_mono(sound_np)
    # else:
    #     sound_data = sound_np[0]
    sound_data = sound
    if train:
        sound_data = augment(samples=sound_data, sample_rate=SAMPLE_RATE)
    # sound_data = mixer(augmented)
    # downsample the audio to ~8kHz
    sound_data = torch.from_numpy(sound_data.reshape((sound_data.shape[0], 1)))
    tempData = torch.zeros([160000, 1])  # tempData accounts for audio clips that are too short
    if sound_data.numel() < 160000:
        tempData[:sound_data.numel()] = sound_data[:]
    else:
        tempData[:] = sound_data[:160000]

    sound_data = tempData
    soundFormatted = torch.zeros([32000, 1])
    # soundFormatted[:32000] = sound_data[::5]  # take every fifth sample of sound_data
    soundFormatted[:32000] = sound_data[::freq]
    soundFormatted = soundFormatted.permute(1, 0)
    return soundFormatted


def process_sound(sound_data, freq=DEF_FREQ, train=True):
    if train:
        sound_data = augment(samples=sound_data, sample_rate=SAMPLE_RATE)
    # sound_data = mixer(augmented)
    # downsample the audio to ~8kHz
    sound_data = torch.from_numpy(sound_data.reshape((sound_data.shape[0], 1)))
    tempData = torch.zeros([160000, 1])  # tempData accounts for audio clips that are too short
    if sound_data.numel() < 160000:
        tempData[:sound_data.numel()] = sound_data[:]
    else:
        tempData[:] = sound_data[:160000]

    sound_data = tempData
    soundFormatted = torch.zeros([32000, 1])
    # soundFormatted[:32000] = sound_data[::5]  # take every fifth sample of sound_data
    soundFormatted[:32000] = sound_data[::freq]
    soundFormatted = soundFormatted.permute(1, 0)
    return soundFormatted
