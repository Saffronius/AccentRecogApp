# AccentRecogApp

AccentRecogApp provides a simple accent recognition pipeline built on top of a
pretrained [Wav2Vec2](https://arxiv.org/abs/2006.11477) model from
`torchaudio`. Audio files are expected to be organised in subfolders named after
the accent label (e.g. `american/`, `british/`).

The training script freezes the pretrained feature extractor and trains a small
classification head for the provided accents.

## Requirements

* Python 3.8+
* `torch`
* `torchaudio`
* `flask` (for the optional web interface)

Install the dependencies with:

```bash
pip install torch torchaudio flask
```

## Training

Prepare a dataset directory where each accent has its own folder containing
`.wav` files. For example:

```
/your-dataset/
    american/
        sample1.wav
        sample2.wav
    british/
        example1.wav
```

Run the training script:

```bash
python -m accentrecog.train /your-dataset --epochs 5 --output model.pt
```

This will save `model.pt` containing the trained weights and label mapping.

## Prediction

After training you can predict the accent of a new `.wav` file:

```bash
python -m accentrecog.predict model.pt path/to/test.wav
```

The script prints the predicted accent label.

## Web Interface

You can start a simple web frontend to upload audio files and get
predictions using `Flask`:

```bash
python webapp.py
```

By default the application looks for a model file named `accent_model.pt` in the
current directory. Set the `MODEL_PATH` environment variable to point to a
different model if needed.

## Project Ideas

* Collect a diverse dataset of speakers from different regions to improve the
  model's robustness.
* Experiment with other pretrained speech models or fine-tune additional layers
  for higher accuracy.
* Build a simple web interface using `Flask` or `FastAPI` so users can upload
  audio and get real‑time accent predictions.
