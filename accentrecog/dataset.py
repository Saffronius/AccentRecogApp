import os
import torchaudio
from torch.utils.data import Dataset

class AccentDataset(Dataset):
    """Dataset that loads audio files organized by accent labels."""

    def __init__(self, root_dir: str, sample_rate: int = 16000):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.files = []
        self.labels = []
        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path):
                continue
            for fname in os.listdir(label_path):
                if fname.lower().endswith('.wav'):
                    self.files.append(os.path.join(label_path, fname))
                    self.labels.append(label)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        label = self.label_to_idx[self.labels[idx]]
        return waveform.squeeze(0), label
