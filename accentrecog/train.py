import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .dataset import AccentDataset
from .model import Wav2Vec2Classifier


def collate_fn(batch):
    waveforms, labels = zip(*batch)
    waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    labels = torch.tensor(labels)
    return waveforms, labels


def train(args):
    dataset = AccentDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = Wav2Vec2Classifier(len(dataset.label_to_idx))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for waveforms, labels in loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}: loss {avg_loss:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'label_to_idx': dataset.label_to_idx,
    }, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train accent recognition model")
    parser.add_argument("data_dir", help="Directory with accent-labeled subfolders")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="accent_model.pt")
    args = parser.parse_args()
    train(args)
