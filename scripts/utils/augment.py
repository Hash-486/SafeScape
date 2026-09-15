"""Waveform-level augmentation: pitch-shift, time-stretch, additive noise, mixup."""
import numpy as np
import librosa

SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE  # 1.0s


def _fix_length(y, n=WINDOW_SAMPLES):
    if len(y) < n:
        return np.pad(y, (0, n - len(y)))
    return y[:n]


def pitch_shift(y, sr=SAMPLE_RATE, max_steps=3.0):
    steps = np.random.uniform(-max_steps, max_steps)
    return _fix_length(librosa.effects.pitch_shift(y, sr=sr, n_steps=steps))


def time_stretch(y, min_rate=0.85, max_rate=1.15):
    rate = np.random.uniform(min_rate, max_rate)
    return _fix_length(librosa.effects.time_stretch(y, rate=rate))


def add_noise(y, min_snr_db=5, max_snr_db=20):
    snr_db = np.random.uniform(min_snr_db, max_snr_db)
    sig_power = np.mean(y ** 2) + 1e-10
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=len(y)).astype(np.float32)
    return (y + noise).astype(np.float32)


def mixup(y1, y2, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    return (lam * y1 + (1 - lam) * y2).astype(np.float32), lam


def spec_augment(feat, freq_mask_param=6, time_mask_param=12, num_masks=2):
    """SpecAugment-style time/frequency masking directly on a cached (F, T) feature
    array. Cheap (pure numpy, no DSP) vs. waveform-domain pitch-shift/time-stretch,
    which is why it's used for the precomputed-feature training pipeline."""
    feat = feat.copy()
    n_mels, n_frames = feat.shape
    fill = feat.mean()
    for _ in range(num_masks):
        f = np.random.randint(0, min(freq_mask_param, n_mels) + 1)
        if f > 0:
            f0 = np.random.randint(0, n_mels - f + 1)
            feat[f0:f0 + f, :] = fill
        t = np.random.randint(0, min(time_mask_param, n_frames) + 1)
        if t > 0:
            t0 = np.random.randint(0, n_frames - t + 1)
            feat[:, t0:t0 + t] = fill
    return feat


def add_feature_noise(feat, std=0.15):
    """Cheap additive Gaussian noise in the (already normalized-ish) feature domain."""
    return feat + np.random.normal(0, std, size=feat.shape).astype(np.float32)


def augment_feature(feat, p=0.6):
    """Fast augmentation applied to a cached MFCC/log-mel array (replaces
    waveform-domain augment_waveform for the precomputed-feature pipeline)."""
    if np.random.rand() > p:
        return feat
    feat = spec_augment(feat)
    if np.random.rand() < 0.5:
        feat = add_feature_noise(feat)
    return feat.astype(np.float32)


def augment_waveform(y, sr=SAMPLE_RATE, p=0.6):
    """Apply a random subset of augmentations with overall probability p."""
    if np.random.rand() > p:
        return y
    choices = np.random.choice(["pitch", "stretch", "noise"], size=np.random.randint(1, 3), replace=False)
    out = y
    for c in choices:
        if c == "pitch":
            out = pitch_shift(out, sr)
        elif c == "stretch":
            out = time_stretch(out)
        elif c == "noise":
            out = add_noise(out)
    return out.astype(np.float32)
