import librosa
import numpy as np
import torch
from config import SAMPLE_RATE


def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    wav, sr = librosa.load(file_path, sr=sr)
    if wav.shape[0] < 5 * sr:
        wav = np.pad(wav, int(np.ceil((5 * sr - wav.shape[0]) / 2)), mode='reflect')
    else:
        wav = wav[:5 * sr]
    spec = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db


def get_samples_from_file(npy_file_path, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300):
    wave = np.load(npy_file_path)
    sample_length = 5 * wave
    result = []
    for idx in range(0, len(wave), sample_length):
        cropped_wave = wave[idx:idx + sample_length]
        if len(cropped_wave) < sample_length:
            cropped_wave = np.pad(cropped_wave, int(np.ceil((5 * SAMPLE_RATE - cropped_wave.shape[0]) / 2)),
                                  mode='reflect')
        spec = librosa.feature.melspectrogram(cropped_wave, sr=SAMPLE_RATE, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
        result.append(torch.from_numpy(spec))
    return result
