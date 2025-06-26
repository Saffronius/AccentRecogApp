import torch
from torch import nn
import torchaudio

class Wav2Vec2Classifier(nn.Module):
    """Classifier built on top of a pretrained Wav2Vec2 model."""

    def __init__(self, num_classes: int):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.feature_extractor = bundle.get_model()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(bundle._params["encoder_embed_dim"], 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            features, _ = self.feature_extractor.extract_features(waveforms)
        x = features[-1].mean(dim=1)
        return self.classifier(x)
