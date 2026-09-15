"""Model architectures: MFCC-CNN, log-mel CRNN, and a stretch-goal tiny transformer."""
import torch
import torch.nn as nn

NUM_CLASSES = 5


class MFCCCNN(nn.Module):
    """Compact 2D CNN over MFCC time-frequency images."""

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, n_mfcc, T)
        return self.classifier(self.features(x))


class LogMelCRNN(nn.Module):
    """CNN feature extractor over log-mel spectrograms + BiGRU temporal head."""

    def __init__(self, num_classes=NUM_CLASSES, hidden_size=64, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )
        # after 2x freq-pooling: n_mels=64 -> 16; channels=32 -> feature dim = 32*16 = 512
        self.gru = nn.GRU(input_size=32 * 16, hidden_size=hidden_size, num_layers=1,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        c = self.conv(x)  # (B, C, F', T)
        b, ch, f, t = c.shape
        c = c.permute(0, 3, 1, 2).reshape(b, t, ch * f)  # (B, T, C*F')
        out, _ = self.gru(c)  # (B, T, 2*hidden)
        pooled = out.mean(dim=1)  # mean-pool over time
        return self.fc(self.dropout(pooled))


class TinyTransformer(nn.Module):
    """Stretch-goal: tiny transformer encoder over patchified log-mel spectrogram,
    intended to be trained via knowledge distillation from LogMelCRNN."""

    def __init__(self, num_classes=NUM_CLASSES, n_mels=64, d_model=64, n_layers=2, n_heads=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 200, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, 1, n_mels, T) -> treat each time frame as a token
        x = x.squeeze(1).transpose(1, 2)  # (B, T, n_mels)
        b, t, _ = x.shape
        tok = self.input_proj(x)  # (B, T, d_model)
        cls = self.cls_token.expand(b, -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        tok = tok + self.pos_embed[:, : tok.size(1), :]
        enc = self.encoder(tok)
        return self.fc(enc[:, 0])


def build_model(name, **kwargs):
    if name == "mfcc_cnn":
        return MFCCCNN(**kwargs)
    if name == "logmel_crnn":
        return LogMelCRNN(**kwargs)
    if name == "transformer":
        return TinyTransformer(**kwargs)
    raise ValueError(f"unknown model name: {name}")


if __name__ == "__main__":
    for name, shape in [("mfcc_cnn", (2, 1, 40, 101)), ("logmel_crnn", (2, 1, 64, 101)), ("transformer", (2, 1, 64, 101))]:
        m = build_model(name)
        x = torch.randn(*shape)
        y = m(x)
        loss = y.sum()
        loss.backward()
        n_params = sum(p.numel() for p in m.parameters())
        print(name, "output", tuple(y.shape), "params", n_params)
