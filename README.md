# DSAN 6600 Project — Music Information Retrieval

Deep learning on the **Lakh MIDI Dataset (LMD full)** (~174K MIDI files, [colinraffel.com/projects/lmd](https://colinraffel.com/projects/lmd/)) for two tasks:

- **Classification** — predict key signature (10 major keys) from piano-roll
- **Regression** — predict initial BPM from piano-roll

Training runs on **Google Colab** (NVIDIA A100 / L4).

## Repository Layout

```
scripts/
  data-processing/
    process_midi.ipynb          # Parse LMD → lmd_full_metadata.csv
    midi_to_spectrogram.ipynb   # MIDI → piano-roll .npz + TruncatedSVD
    process_spects.ipynb        # MIDI → mel-spectrogram .npy
  models/
    # Piano-roll input
    DSAN_6600_Piano_Class_CNN.ipynb                # CNN key-signature classifier
    DSAN_6600_Piano_Class_CNN_HP_Tuning.ipynb      # CNN classification HP search
    DSAN_6600_Piano_Class_CNN_GRU.ipynb            # CNN+BiGRU key-signature classifier
    DSAN_6600_Piano_Class_CNN_GRU_HP_Tuning.ipynb  # CNN+BiGRU classification HP search
    DSAN_6600_Piano_Regr_CNN_HP_Tuning.ipynb       # CNN BPM regression HP search
    DSAN_6600_Piano_Regr_CNN_GRU_HP_Tuning.ipynb   # CNN+BiGRU BPM regression HP search
    # Mel-spectrogram input
    DSAN_6600_Spect_Class_CNN.ipynb                # CNN key-signature classifier (spectrogram)
    DSAN_6600_Spect_Class_CNN_HP_tuning.ipynb      # Spectrogram classification HP search
    DSAN_6600_Spect_Regr_CNN.ipynb                 # CNN BPM regression (spectrogram)
data/
  processed_data/
    lmd_full_metadata.csv       # only tracked data file
```

## Data Pipeline

1. **`process_midi.ipynb`** — walks `data/raw_data/lmd_full/`, extracts MIDI metadata with `pretty_midi`, writes `lmd_full_metadata.csv` (~155K rows). Read CSV with `parse_dates=False` (e.g. `3/4` parses as a date otherwise).
2. **`midi_to_spectrogram.ipynb`** — converts MIDIs to piano rolls at 31.25 fps, saves `(128, T)` uint8 `.npz`. Also fits `TruncatedSVD(32)` for reduced `(32, T)` rolls.
3. **`process_spects.ipynb`** (alt) — synthesizes MIDI via FluidSynth (`piano.sf2` required), computes 128-bin mel spectrograms (16 kHz, 15 s).

## Input Representation

Piano rolls as `(1, 128, 468)` float32 tensors — 128 pitch bins × 468 time frames (15 s × 31.25 fps). Loaded as uint8, normalized to `[0, 1]` via `/ 127.0`.

## Models

- **`GeneralCNN`** — 4 Conv-BN-ReLU-Pool blocks (1→32→64→128→256) + 3-layer MLP. ~78% test accuracy.
- **`HybridCNN`** — same CNN backbone + bidirectional GRU (hidden 512) + MLP. ~80% classification accuracy; also used for regression.
- **`SpectrogramCNNDynamic`** — parameterized CNN (3–5 layers) + configurable MLP, used for BPM regression HP search.

## Training

- PyTorch with `torch.amp` mixed precision
- Splits: classification 70/15/15 stratified; regression 80/10/10
- BPM targets z-score normalized with training-set stats (mean ≈ 114, std ≈ 33)
- C major capped at 4500 samples; classes <1000 dropped → 10 classes, ~37K samples
- Early stopping (patience 3–4); checkpoints saved to Google Drive

## Dependencies

`pretty_midi`, `librosa`, `mir_eval`, `torch`, `torchvision`, `sklearn`, `joblib`, `tqdm`
