"""MFCC and log-mel spectrogram feature extraction."""
import numpy as np
import librosa

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MFCC = 40
N_MELS = 64


def mfcc_features(y, sr=SAMPLE_RATE):
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return m.astype(np.float32)  # (N_MFCC, T)


def logmel_features(y, sr=SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel.astype(np.float32)  # (N_MELS, T)


def normalize(feat):
    mean = feat.mean()
    std = feat.std() + 1e-6
    return (feat - mean) / std
