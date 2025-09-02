TOOL SETUP:
    1. python -m pip install --upgrade pip wheel setuptools
    2. pip install numpy scipy librosa==0.10.2.post1 soundfile matplotlib pandas

OVERVIEW:
    This project extracts audio features associated with rhythm using Python, NumPy, and Librosa.
    The processing pipeline accepts audio file as .mp3, .wav, .flac, or .m4a files and outputs:
        1. A CSV file with one row per audio file, containing filename and extracted features.
        2. Plots: spectrogram, onset envelope, and tempogram for each audio file.

FEATURES EXTRACTED:
    1. Energy: RMS, Zero-Crossing Rate.
    2. Spectral: Centroid, Bandwidth, Rolloff.
    3. Timbre: 13 dimensional MFCCs (mean and std).
    4. Harmony: Chroma features (mean and std).
    5. Onsets: Onset envelope stats, number of onset and rate.
    6. Beat and Tempo: BPM, number of beat, inter-beat intervals (IBI).
    7. Tempogram: mean and std across the tempo candidates.
    8. Pitch: F0 median/mean/std, voiced ratio.


PIPELINE DESIGN:
    1. Load audio: waveform normalized to [-1,1].
    2. Extract features: spectral, rhythmic, and pitch statistics.
    3. Save plots: spectrogram, onset envelope, tempogram.
    4. Batch process: combine rows into a CSV.


DESIGN:

CLI (argparse) main()
        |
        |
        v
 run_pipeline()
        |
        |
   +----+-------+
   | Directory? |
   +----+-------+
        |yes
        |
        v
 iterate files
        |
        |
        v
 featurize_file()  <--- single file also goes here
        |
        +--> load_audio()  --> waveform y, sampling rate sr (normalized)
        |
        +--> extract_features()
        |       - Energy: RMS, ZCR
        |       - Spectral: centroid, bandwidth, rolloff
        |       - Timbre: MFCC(13) mean/std
        |       - Harmony: Chroma(12) mean/std
        |       - Onsets: env mean/std, count, rate
        |       - Beat & Tempo: BPM, beat_count, IBI stats
        |       - Tempogram mean/std
        |       - Pitch: f0 stats, voiced_ratio
        |
        +--> if plots enabled:
                 save_plots(): spectrogram, onset envelope, tempogram
        |
        +--> return {file_name, features...}
                |
                v
        accumulate rows → DataFrame → CSV (outputs/features.csv)

OUTPUT: All outputs saved into the "outputs" folder.

TO RUN:
python main.py data --csv outputs/features.csv