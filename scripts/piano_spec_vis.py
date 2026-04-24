import os
import glob
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
input_dir    = "../data/raw_data/lmd_full/"
output_dir   = "../outputs/"
sf2_path     = "../data/raw_data/piano.sf2"
duration     = 15
sr           = 16000
n_mels       = 128
piano_roll_fs = 31.25
target_samples = duration * sr
target_frames  = int(duration * piano_roll_fs)  # 468

os.makedirs(output_dir, exist_ok=True)

# --- Find first MIDI file ---
midi_files = glob.glob(os.path.join(input_dir, "**/*.mid"), recursive=True)
if not midi_files:
    raise FileNotFoundError(f"No .mid files found in {input_dir}")

file_path = midi_files[0]
print(f"Using: {file_path}")

midi_data = pretty_midi.PrettyMIDI(file_path)

# --- Piano Roll ---
roll = midi_data.get_piano_roll(fs=piano_roll_fs)       # (128, T)
roll = roll[:, :target_frames]                           # truncate to 15s
if roll.shape[1] < target_frames:
    pad = np.zeros((128, target_frames - roll.shape[1]))
    roll = np.hstack((roll, pad))
roll = roll.astype(np.uint8)

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(roll, aspect='auto', origin='lower', cmap='gray_r')
ax.set_title('Piano Roll')
ax.set_xlabel('Time Frames')
ax.set_ylabel('Pitch')
ax.set_yticks(range(0, 128, 12))
ax.set_yticklabels([f'C{i}' for i in range(11)])
plt.colorbar(ax.images[0], ax=ax, label='Velocity (0–127)')
plt.tight_layout()
piano_roll_path = os.path.join(output_dir, 'piano_roll_example.png')
plt.savefig(piano_roll_path, dpi=150)
plt.close()
print(f"Saved piano roll to {piano_roll_path}")

# --- Spectrogram ---
for instrument in midi_data.instruments:
    instrument.program = 0
    instrument.is_drum = False

y = midi_data.fluidsynth(fs=sr, sf2_path=sf2_path)

if len(y) >= target_samples:
    y = y[:target_samples]
else:
    y = np.concatenate((y, np.zeros(target_samples - len(y))))

S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmin=20, fmax=5000)
S_db = librosa.power_to_db(S, ref=np.max)

fig, ax = plt.subplots(figsize=(8, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel',
                         fmin=20, fmax=5000, cmap='magma', ax=ax)
ax.set_title('Mel Spectrogram')
plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
plt.tight_layout()
spectrogram_path = os.path.join(output_dir, 'spectrogram_example.png')
plt.savefig(spectrogram_path, dpi=150)
plt.close()
print(f"Saved spectrogram to {spectrogram_path}")

print("Done!")