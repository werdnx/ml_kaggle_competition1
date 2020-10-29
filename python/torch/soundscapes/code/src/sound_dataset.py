import os

import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

# A,B,C,D,E,F,G,H,I
CATEGORIES = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}


class SoundDataset(Dataset):
    # rapper for the UrbanSound8K dataset
    # Argument List
    #  path to the UrbanSound8K csv file
    #  path to the UrbanSound8K audio files
    #  list of folders to use in the dataset

    def __init__(self, base, df):
        self.df = df
        self.base = base
        self.labels = []
        for ind in tqdm(range(len(df))):
            row = df.iloc[ind]
            self.labels.append(CATEGORIES[row[1]])
        self.mixer = torchaudio.transforms.DownmixMono()

    def __getitem__(self, index):
        # format the file path and load the file
        row = self.df.iloc[index]
        path = os.path.join(self.base, row[0])
        path = path + '.wav'
        sound = torchaudio.load(path, out=None, normalization=True)
        # load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
        soundData = self.mixer(sound[0])
        # downsample the audio to ~8kHz
        tempData = torch.zeros([160000, 1])  # tempData accounts for audio clips that are too short
        if soundData.numel() < 160000:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:160000]

        soundData = tempData
        soundFormatted = torch.zeros([32000, 1])
        soundFormatted[:32000] = soundData[::5]  # take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        return soundFormatted, self.labels[index]

    def __len__(self):
        return len(self.file_names)
