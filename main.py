"""
Rhythm feature pipeline.

Input:  file path OR directory of .mp3/.wav files
Output: CSV with one row per file.
Also saves helpful plots (spectrogram, tempogram, onset envelope) into outputs folder.

Covers:
1) Loading audio
2) Feature extraction
3) Visualization
4) Time–frequency analysis
5) Pitch estimation
6) Beat & tempo analysis
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from librosa import feature, onset, beat, display




# 1) Loading audio

def load_audio(path: Path, sr: int = 22050, mono: bool = True, duration: float | None = None):
    """
    Load an audio file (mp3/wav).
    sr: target sample rate (22050 is a common default for Musical Information Retrival tasks)
    mono: mix down to monophonic for consistency and speed
    duration: If set, only load the first N seconds
    """
    # librosa.load returns audio time series and sampling rate
    y, sr = librosa.load(str(path), sr=sr, mono=mono, duration=duration) # y = waveform, sr = sampling rate

    # Normalize waveform to [-1, 1] so all files are on the same scale
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y, sr  #return normalized waveform and sampling rate


# Helpers

def mean_std(x: np.ndarray) -> Tuple[float, float]:

    #compute mean and std deviation of 1D vector
    x = np.asarray(x).ravel() # flatten to 1D
    return float(np.nanmean(x)), float(np.nanstd(x)) #ignore Nan values

def summarize_vector(vec: np.ndarray, prefix: str, agg=("mean","std")) -> Dict[str, float]:
    """
       Summarize multidimensional features (like MFCCs, chroma).
       Returns dictionary with mean & std for each coefficient.
    """
    stats = {}
    vec = np.asarray(vec)
    if vec.ndim == 1:   #if handed a single vector (T,), treat it as 1 row of length T.
        vec = vec[None, :]
    for i, row in enumerate(vec): #loop over feature coefficients
        m, s = mean_std(row)
        if "mean" in agg: stats[f"{prefix}{i+1}_mean"] = m
        if "std"  in agg: stats[f"{prefix}{i+1}_std"]  = s
    return stats


# 2) Feature Extraction

"""
Compute a compact set of features that capture loudness, brightness,
    timbre, harmony, rhythmic activity, tempo, and pitch.
"""
def extract_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    feats: Dict[str, float] = {}
#Extract rhythm, pitch, spectral, and onset features from audio.

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))  # Short-time Fourier transform (time-frequency representation)

    # Basic spectral/energy features
    rms = librosa.feature.rms(S=S)[0]                       # Root Mean Square energy
    zcr = librosa.feature.zero_crossing_rate(y)[0]          # Zero-crossing rate (noisiness)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]  # Spectral centroid (brightness)
    bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]    # Spectral bandwidth (spread around centroid)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]   # Rolloff 85%, for each frame of the frequency( high enough to reflect brightness)

    # Store audio duration
    feats["duration_sec"] = float(len(y) / sr)

    # Compute mean & std for each feature above
    for name, arr in [("rms",rms), ("zcr",zcr), ("centroid",centroid),
                      ("bandwidth",bw), ("rolloff",rolloff)]:
        m, s = mean_std(arr)
        feats[f"{name}_mean"] = m
        feats[f"{name}_std"]  = s

    # MFCCs (Mel-Frequency Cepstral Coefficients, are capture timbre that often correlates with rhythm)

    # 13 coefficients is a standard setting and summarize each across time
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feats.update(summarize_vector(mfcc, "mfcc"))

    # Chroma (harmonic context can influence perceived rhythm)
    chroma = librosa.feature.chroma_stft(S=S**2, sr=sr)
    feats.update(summarize_vector(chroma, "chroma"))

    # Onset envelope (transient strength) and onset rate
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    feats["onset_env_mean"], feats["onset_env_std"] = mean_std(onset_env)

    # Count number onset rate per second and measure rhythm density
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
    feats["onset_count"] = float(len(onsets))
    feats["onset_rate_per_sec"] = float(len(onsets) / max(feats["duration_sec"], 1e-9))

    #  Beat tracking (Tempo and beat location)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="time")
    feats["tempo_bpm"] = float(tempo)
    feats["beat_count"] = float(len(beat_frames))

    # inter beat intervals quantify tempo stability and NaN if too few beats
    if len(beat_frames) >= 2:
        ibis = np.diff(beat_frames)              # gaps between beats
        feats["ibi_mean_sec"], feats["ibi_std_sec"] = mean_std(ibis)
    else:
        feats["ibi_mean_sec"] = np.nan
        feats["ibi_std_sec"]  = np.nan

    # Tempogram summaries (timefrequency of tempo)
    onsets_env = onset_env
    tg = librosa.feature.tempogram(onset_envelope=onsets_env, sr=sr)  # shape (lags, T)
    feats.update({
        "tempogram_mean": float(np.nanmean(tg)),
        "tempogram_std":  float(np.nanstd(tg)),
    })

    #  Pitch Estimation (robust summary statistics) using probabilistic YIN(pYIN)
    # If it fails, fall back to YIN
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(y,
                                        fmin=librosa.note_to_hz("C2"),
                                        fmax=librosa.note_to_hz("C7"),
                                        sr=sr)
    except Exception:
        # fallback to YIN if pYIN fails or unavailable
        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"),
                         fmax=librosa.note_to_hz("C7"), sr=sr)
        #optional
        voiced_flag = ~np.isnan(f0)
        voiced_prob = None

    # Process of fundamental frequency (f0)
    # Summary of F0 with robust stats and report how much of the clip is voiced.
    if f0 is not None:      # proceed only if YIN returned something
        f0 = np.asarray(f0)
        voiced = f0[~np.isnan(f0)] # keep only voiced frames where pitch was detected
        if voiced.size:
            feats["f0_median_hz"] = float(np.median(voiced))
            feats["f0_mean_hz"]   = float(np.mean(voiced))
            feats["f0_std_hz"]    = float(np.std(voiced))
            feats["voiced_ratio"] = float(voiced.size / f0.size)

        else:                    #ill with NaNs and 0% voiced
            feats.update({"f0_median_hz": np.nan, "f0_mean_hz": np.nan,
                          "f0_std_hz": np.nan, "voiced_ratio": 0.0})
    return feats


# 3) Visualization helpers

def save_plots(y: np.ndarray, sr: int, file_stem: str, out_dir: Path):
    # Save spectrogram, onset envelope, and tempogram plots

    out_dir.mkdir(parents=True, exist_ok=True)

    # Spectrogram shows frequency content over time
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure()
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.title("Log Spectrogram")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_stem}_spectrogram.png", dpi=150)
    plt.close()

    # Onset envelope plot
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    plt.figure()
    plt.plot(times, onset_env)
    plt.title("Onset Envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Strength")
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_stem}_onset_envelope.png", dpi=150)
    plt.close()

    # Tempogram plot
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    plt.figure()
    librosa.display.specshow(tg, x_axis="time", y_axis="tempo", sr=sr)
    plt.title("Tempogram")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_stem}_tempogram.png", dpi=150)
    plt.close()


# 4) One-file featurization that returns Name and Vector

def featurize_file(path: Path, duration: float | None = 30.0,
                   make_plots: bool = True, plots_dir: Path = Path("outputs")) -> Tuple[str, Dict[str, float]]:
    # Extract features and save plots for a single audio file

    y, sr = load_audio(path, sr=22050, mono=True, duration=duration)
    feats = extract_features(y, sr)
    if make_plots:
        save_plots(y, sr, path.stem, plots_dir)
    return path.name, feats


# 5) Batch pipeline (directory or single file) to CSV

def run_pipeline(input_path: Path, out_csv: Path, duration: float | None = 30.0, plots: bool = True):
    # Process one or many audio files, save features to CSV
    files: List[Path] = []
    if input_path.is_dir():  # handle multiple files
        for ext in (".mp3", ".wav", ".flac", ".m4a"):
            files.extend(sorted(input_path.rglob(f"*{ext}")))
    else:    # single file
        files = [input_path]

    rows = []
    feature_keys = None

    for f in files:
        print(f"Processing {f.name} ...")
        name, feats = featurize_file(f, duration=duration, make_plots=plots, plots_dir=Path("outputs"))

        # Initialize the header order
        if feature_keys is None:
            feature_keys = ["file_name"] + sorted(feats.keys())
        # Build row with file name and features
        row = {"file_name": name}
        row.update(feats)
        rows.append(row)

    # Build DataFrame
    df = pd.DataFrame(rows)
    # Reorder columns with file_name first, remaining sorted
    cols = ["file_name"] + sorted([c for c in df.columns if c != "file_name"])
    df = df[cols]

    # save CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved features to: {out_csv.resolve()}")
    print("Plots (per file) saved to: outputs/")


# CLI

def main():
    # command line interface entrypoint
    p = argparse.ArgumentParser(description="Rhythm feature extraction pipeline")
    p.add_argument("input", type=str, help="Path to audio file or directory")
    p.add_argument("--csv", type=str, default="outputs/features.csv", help="Output CSV path")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Seconds to analyze from start (None = full file)")
    p.add_argument("--no-plots", action="store_true", help="Disable saving plots")
    args = p.parse_args()

    in_path = Path(args.input)
    run_pipeline(in_path, Path(args.csv),
                 duration=None if args.duration in (None, 0) else args.duration,
                 plots=not args.no_plots)

if __name__ == "__main__":
    main()
