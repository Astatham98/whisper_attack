#!/usr/bin/env python3
"""
Compute Mean Squared Error (MSE) between two audio files.

Usage:
    python audio_mse.py file1.wav file2.wav

Requirements:
    pip install librosa numpy
"""

import sys
import numpy as np
import librosa

def compute_mse(audio1, audio2, sr1, sr2):
    """Compute MSE between two audio signals, handling sample rate and length differences."""

    # Resample if sample rates differ
    if sr1 != sr2:
        print(f"Warning: Sample rates differ ({sr1} Hz vs {sr2} Hz). Resampling to {sr1} Hz.")
        audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
        sr2 = sr1

    # Ensure mono by averaging channels if stereo
    if len(audio1.shape) > 1:
        audio1 = np.mean(audio1, axis=0)
    if len(audio2.shape) > 1:
        audio2 = np.mean(audio2, axis=0)

    # Trim or pad to match lengths
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]

    if len(audio1) != len(audio2):
        print("Warning: Audio files were trimmed to match the shortest length.")

    # Compute MSE
    mse = np.mean((audio1 - audio2) ** 2)
    return mse

def main():
    if len(sys.argv) != 3:
        print("Usage: python audio_mse.py <file1> <file2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    try:
        # Load audio files
        audio1, sr1 = librosa.load(file1, sr=None)
        audio2, sr2 = librosa.load(file2, sr=None)

        # Compute MSE
        mse = compute_mse(audio1, audio2, sr1, sr2)

        # Output result
        print(f"\nMSE between '{file1}' and '{file2}': {mse:.10f}")
        print(f"MSE (dB): {10 * np.log10(mse):.2f}" if mse > 0 else "MSE (dB): -inf")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()