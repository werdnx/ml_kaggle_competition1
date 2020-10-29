import torch
import torchaudio

mixer = torchaudio.transforms.DownmixMono()


def process_file(file_path, freq):
    sound = torchaudio.load(file_path, out=None, normalization=True)
    # load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
    soundData = mixer(sound[0])
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
