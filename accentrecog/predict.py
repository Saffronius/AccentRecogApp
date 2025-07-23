import argparse
import torch
import torchaudio

from .model import Wav2Vec2Classifier


def predict_file(model_path: str, audio_path: str) -> str:
    """Return the predicted accent label for ``audio_path`` using ``model_path``."""
    checkpoint = torch.load(model_path, map_location="cpu")
    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    model = Wav2Vec2Classifier(len(label_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    with torch.no_grad():
        output = model(waveform)
        pred = output.argmax(dim=-1).item()

    return idx_to_label[pred]


def predict(args):
    """CLI entry point."""
    label = predict_file(args.model, args.audio)
    print(label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict accent of an audio file")
    parser.add_argument("model", help="Path to trained model file")
    parser.add_argument("audio", help="Audio file (.wav) to classify")
    args = parser.parse_args()
    predict(args)
