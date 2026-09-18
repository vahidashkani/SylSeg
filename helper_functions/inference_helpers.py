#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:38:00 2026

@author: vahid
"""

import os
import sys
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from transformers import AutoFeatureExtractor, WhisperModel
# IMPORT MODEL
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model.model import GRBASPredictor


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# set some variables and parameters
SAMPLE_RATE = 16000
N_FFT = 400
WIN_LENGTH = 400
HOP_LENGTH = 320
CENTER = False

F1_THRESHOLD = 0.5
F1_MIN_DISTANCE = 2

FRAME_MS = 1000.0 * HOP_LENGTH / SAMPLE_RATE

MERGE_ONSET_OFFSET_DISTANCE_MS = 20.0


# AUDIO LOADING
def load_wav_soundfile(wav_path: str):
    """
    Load WAV file using soundfile.

    Returns
    -------
    wav : torch.Tensor
        Shape = [1, samples]

    sr : int
        Sampling rate.
    """

    x, sr = sf.read(wav_path, always_2d=True)
    x = x.astype(np.float32)
    # [samples, channels] -> [channels, samples]
    wav = torch.from_numpy(x.T)
    # Convert stereo/multi-channel to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0,keepdim=True)
    return wav.contiguous(), int(sr)


# FEATURE EXTRACTION
def compute_stft_features(
    wav_1d: torch.Tensor,
    device: torch.device):
    """
    Compute magnitude STFT + delta + delta-delta features.
    """

    window = torch.hann_window(WIN_LENGTH, device=device)

    X = torch.stft(wav_1d, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=CENTER,
        return_complex=True,
        onesided=True,
        pad_mode="reflect",)

    mag = torch.abs(X)
    delta = torchaudio.functional.compute_deltas(mag)
    delta2 = torchaudio.functional.compute_deltas(delta)
    features = torch.cat([mag, delta, delta2], dim=0)
    return features


def frame_mid_time_sec(fi: int) -> float:
    """
    Convert model frame index to time in seconds.
    """
    return (fi * HOP_LENGTH + WIN_LENGTH / 2.0) / float(SAMPLE_RATE)


# =============================================================================
# CHECKPOINT / MODEL LOADING
# =============================================================================
def torch_load_safe(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)

    except TypeError:

        return torch.load(path, map_location="cpu")


def get_state_dict_from_checkpoint(ckpt):

    if (isinstance(ckpt, dict)
        and "model_state_dict" in ckpt
        and isinstance(ckpt["model_state_dict"], dict)):
        return ckpt["model_state_dict"]

    if (isinstance(ckpt, dict)
        and any(isinstance(v, torch.Tensor)
            for v in ckpt.values())):
        return ckpt

    raise RuntimeError(
        "Checkpoint format not recognized.")


def load_model_weights_robust(model: nn.Module,ckpt_obj):
    state_dict = get_state_dict_from_checkpoint(ckpt_obj)

    incompatible = model.load_state_dict(
        state_dict,
        strict=False)

    if (getattr(incompatible, "missing_keys",None)
        and len(incompatible.missing_keys) > 0):
        print("[WARN] Missing keys:")

        for k in incompatible.missing_keys:
            print("   ", k)

    if (getattr(incompatible, "unexpected_keys", None)
        and len(incompatible.unexpected_keys) > 0):
        print("[WARN] Unexpected keys:")

        for k in incompatible.unexpected_keys:
            print("   ", k)


def create_model_for_test(device: torch.device, CHECKPOINT: str):
    """
    Create model and load trained checkpoint.
    """

    print("\nLoading Whisper model...")
    asr_model = WhisperModel.from_pretrained("openai/whisper-tiny", output_attentions=True)

    decoder_input_ids = (torch.tensor([[1, 1]]) * asr_model.config.decoder_start_token_id)

    decoder_input_ids = decoder_input_ids.to(device)
    model = GRBASPredictor(
        asr_model=asr_model,
        decoder_input_ids=decoder_input_ids,
        ssl_out_dim=384,
        dropout_concat=0.40,
        lstm_dropout=0.35,
        local_conv_channels=128,
        lstm_hidden=64,).to(device)

    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch_load_safe(CHECKPOINT)
    load_model_weights_robust(model, ckpt)
    model.eval()

    print("Model loaded successfully.")
    return model


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================
def get_above_threshold_regions(probs: np.ndarray, threshold: float) -> List[Tuple[int, int]]:

    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    T = len(probs)
    regions = []
    i = 0

    while i < T:
        if probs[i] <= float(threshold):
            i += 1
            continue

        start = i
        while (i < T
            and probs[i] > float(threshold)):
            i += 1
        end = i
        if end > start:
            regions.append((int(start), int(end)))
    return regions


def detect_peaks_1d(probs: np.ndarray, threshold: float, min_distance: int) -> List[int]:
    """
    Detect one peak from every contiguous
    above-threshold probability region.
    """

    probs = np.asarray(probs, dtype=np.float32).reshape(-1) 
    peaks = []

    for start, end in get_above_threshold_regions( probs, threshold):
        region = probs[start:end]

        peak = (start + int(np.argmax(region)))

        peaks.append(
            int(peak))
    return peaks


def merge_close_onset_offset_borders(pred_on_frames: List[int], pred_off_frames: List[int], max_distance_ms: float,) -> Tuple[List[int], List[int]]:
    """
    Merge onset and offset borders that are
    very close to each other.
    """

    pred_on_frames = sorted(
        set(int(f) for f in pred_on_frames))

    pred_off_frames = sorted(
        set(int(f) for f in pred_off_frames))

    max_distance_frames = ( float(max_distance_ms) / float(FRAME_MS))
    candidates = []

    for i, on_f in enumerate(
        pred_on_frames):
        for j, off_f in enumerate(
            pred_off_frames):
            dist_frames = abs(
                int(on_f)
                - int(off_f))

            if (dist_frames <= max_distance_frames + 1e-9):

                candidates.append(
                    (dist_frames, i, j, int(on_f), int(off_f)))
    candidates.sort(key=lambda x: x[0])

    used_on = set()
    used_off = set()
    on_replace = {}
    off_replace = {}

    for (_,i, j, on_f, off_f) in candidates:
        if (i in used_on or j in used_off):
            continue

        merged_f = int(
            round((int(on_f) + int(off_f))/ 2.0))

        used_on.add(i)
        used_off.add(j)

        on_replace[i] = merged_f
        off_replace[j] = merged_f

    merged_on_frames = []
    for i, f in enumerate(
        pred_on_frames):
        merged_on_frames.append(int(on_replace.get(i, int(f))))

    merged_off_frames = []
    for j, f in enumerate(
        pred_off_frames):

        merged_off_frames.append(int(off_replace.get(j, int(f))))

    return (sorted(set(merged_on_frames)), sorted(set(merged_off_frames)))


def enforce_onset_offset_order(pred_on_frames: List[int], pred_off_frames: List[int],
    onset_probs: np.ndarray,
    offset_probs: np.ndarray,) -> Tuple[List[int], List[int]]:
    """
    Enforce:

        onset -> offset -> onset -> offset ...

    Double onset:
        keep first onset

    Double offset:
        keep second offset

    Final unmatched onset:
        close at signal end.
    """

    onset_probs = np.asarray(onset_probs, dtype=np.float32).reshape(-1)

    offset_probs = np.asarray( offset_probs, dtype=np.float32).reshape(-1)

    pred_on_frames, pred_off_frames = (merge_close_onset_offset_borders(pred_on_frames=pred_on_frames,
            pred_off_frames=pred_off_frames,
            max_distance_ms=MERGE_ONSET_OFFSET_DISTANCE_MS,))

    events = {}
    for f in pred_on_frames:

        events.setdefault(
            int(f),{"onset": False,
                "offset": False})["onset"] = True

    for f in pred_off_frames:
        events.setdefault(int(f),{"onset": False, "offset": False})["offset"] = True

    out_on = []
    out_off = []
    expected = "onset"

    for f in sorted(events.keys()):
        f = int(f)
        has_on = bool(events[f]["onset"])
        has_off = bool(events[f]["offset"])
        # -----------------------------------------------------
        # Both onset and offset at same frame
        # -----------------------------------------------------
        if has_on and has_off:
            if expected == "onset":
                if (len(out_off) > 0 and len(out_on) == len(out_off)):
                    out_off[-1] = f
                out_on.append(f)
                expected = "offset"
            else:
                out_off.append(f)
                out_on.append(f)
                expected = "offset"
            continue

        # -----------------------------------------------------
        # Onset only
        # -----------------------------------------------------
        if has_on:

            if expected == "onset":
                out_on.append(f)
                expected = "offset"
            else:
                # Consecutive onset:
                # keep first onset
                pass
            continue

        # -----------------------------------------------------
        # Offset only
        # -----------------------------------------------------
        if has_off:
            if expected == "offset":
                out_off.append(f)
                expected = "onset"
            else:

                # Consecutive offset:
                # replace previous offset
                # with current/latest offset.
                if len(out_off) > 0:
                    out_off[-1] = f
                expected = "onset"
            continue

    T = int(min(len(onset_probs), len(offset_probs)))
    end_signal_frame = int(T)
    # Final unmatched onset
    while len(out_on) > len(out_off):
        out_off.append(end_signal_frame)
    # Safety
    while len(out_off) > len(out_on):
        out_off.pop(0)
    out_on = sorted(
        int(f)
        for f in out_on)
    out_off = sorted(
        int(f)
        for f in out_off)
    return (out_on, out_off)

def frames_to_times(frames: List[int]) -> List[float]:

    return [frame_mid_time_sec(int(f))
        for f in frames]


# =============================================================================
# SINGLE-WAV INFERENCE
# =============================================================================
def detect_onsets_offsets_from_wav(WAV_PATH: str, model, feature_extractor, device: torch.device,):
    """
    Run onset/offset detection for ONE WAV file.

    IMPORTANT:
    The model is NOT loaded here.

    The model is loaded once before batch
    processing and reused for every WAV.
    """

    wav, sr = load_wav_soundfile(
        WAV_PATH)
    # ---------------------------------------------------------
    # Resample if required
    # ---------------------------------------------------------
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    wav = wav.contiguous()
    wav_1d = wav.squeeze(0)
    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.empty_cache()
        stft_feat = compute_stft_features(wav_1d.to(device),device=device)
        asr_feat = feature_extractor(wav_1d.cpu().numpy().copy(), return_tensors="pt", sampling_rate=SAMPLE_RATE,).input_features
        asr_in = (asr_feat.unsqueeze(0).to(device))

        wav_in = (wav
            .unsqueeze(0)
            .to(device))

        stft_in = (stft_feat
            .unsqueeze(0)
            .to(device))

        onset_logits, offset_logits = model(asr_in,
            wav_in,
            stft_in)

        onset_probs = (torch.sigmoid(onset_logits)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy())

        offset_probs = (torch.sigmoid(offset_logits)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy())

        # -----------------------------------------------------
        # Detect onset peaks
        # -----------------------------------------------------
        pred_on_frames = detect_peaks_1d(onset_probs, threshold=F1_THRESHOLD, min_distance=F1_MIN_DISTANCE,)

        # -----------------------------------------------------
        # Detect offset peaks
        # -----------------------------------------------------
        pred_off_frames = detect_peaks_1d(offset_probs, threshold=F1_THRESHOLD, min_distance=F1_MIN_DISTANCE,)

        # -----------------------------------------------------
        # Enforce onset/offset ordering
        # -----------------------------------------------------
        (pred_on_frames, pred_off_frames) = enforce_onset_offset_order(pred_on_frames=pred_on_frames, pred_off_frames=pred_off_frames,
            onset_probs=onset_probs, offset_probs=offset_probs,)

        prob_T = int(min(len(onset_probs), len(offset_probs)))

        signal_end_time_sec = (float(wav_1d.numel()) / float(SAMPLE_RATE))

        # -----------------------------------------------------
        # Frames -> seconds
        # -----------------------------------------------------
        onset_times = frames_to_times(
            pred_on_frames)

        offset_times = [signal_end_time_sec if int(f) >= prob_T
            else frame_mid_time_sec(
                int(f))
            for f in pred_off_frames]

    onset_times = np.asarray(onset_times, dtype=np.float32)

    offset_times = np.asarray(offset_times, dtype=np.float32)

    return (onset_times, offset_times)


# =============================================================================
# INPUT TSV VALIDATION
# =============================================================================
def load_input_tsv(input_tsv_path: str) -> pd.DataFrame:
    """
    Load and validate subject input TSV.
    """
    if not os.path.isfile(
        input_tsv_path):
        raise FileNotFoundError(
            f"Input TSV does not exist:\n"
            f"{input_tsv_path}")

    df = pd.read_csv(input_tsv_path, sep="\t")

    required_columns = ["subject", "run", "trial", "syllable", "wav_file",]
    missing_columns = [column
        for column in required_columns
        if column not in df.columns]
    if missing_columns:
        raise ValueError("Input TSV is missing required columns: " + ", ".join(missing_columns))

    if len(df) == 0:
        raise ValueError(
            f"Input TSV is empty:\n"
            f"{input_tsv_path}")
    return df






