#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:38:00 2026

@author: vahid
"""

import os
import pandas as pd
import torch
from transformers import AutoFeatureExtractor

from helpers_functions.inference_helpers import (
    create_model_for_test,
    detect_onsets_offsets_from_wav, load_input_tsv,)

# =============================================================================
def process_subject(WAV_FOLDER: str, INPUT_TSV_PATH: str, OUTPUT_FOLDER: str, CHECKPOINT: str,):
    """
    Process ALL WAV files listed in one subject's TSV.

    For each trial:

        onset_times[0] + offset_times[0]
            -> syllable_number = 1

        onset_times[1] + offset_times[1]
            -> syllable_number = 2

        ...

    Output TSV columns:

        subject
        run
        trial
        syllable
        wav_file
        syllable_number
        onset_sec
        offset_sec
    """

    # ---------------------------------------------------------
    # Validate paths
    # ---------------------------------------------------------
    if not os.path.isdir(
        WAV_FOLDER):

        raise NotADirectoryError(
            f"WAV folder does not exist:\n"
            f"{WAV_FOLDER}")

    if not os.path.isfile(
        CHECKPOINT):

        raise FileNotFoundError(
            f"Checkpoint does not exist:\n"
            f"{CHECKPOINT}")

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True)

    # ---------------------------------------------------------
    # Load TSV
    # ---------------------------------------------------------
    input_df = load_input_tsv(
        INPUT_TSV_PATH)

    # ---------------------------------------------------------
    # Determine subject
    # ---------------------------------------------------------
    subjects = (
        input_df["subject"]
        .astype(str)
        .unique()
        .tolist())

    if len(subjects) != 1:

        raise ValueError(
            "The input TSV must contain exactly "
            "one subject.\n"
            f"Found subjects: {subjects}")

    subject_name = subjects[0]

    print("\n" + "=" * 80)
    print(f"SUBJECT: {subject_name}")
    print(f"Number of trials: {len(input_df)}")
    print("=" * 80)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu")

    print(f"DEVICE: {device}")

    # ---------------------------------------------------------
    # Load feature extractor ONCE
    # ---------------------------------------------------------
    print(
        "\nLoading feature extractor...")

    feature_extractor = (
        AutoFeatureExtractor.from_pretrained("openai/whisper-tiny"))

    # ---------------------------------------------------------
    # Load model ONCE
    # ---------------------------------------------------------
    model = create_model_for_test(
        device=device,
        CHECKPOINT=CHECKPOINT)

    # ---------------------------------------------------------
    # Output rows
    # ---------------------------------------------------------
    output_rows = []
    failed_files = []
    total_trials = len(input_df)

    # ---------------------------------------------------------
    # Build lookup from INPUT_TSV_PATH:
    #
    # wav_file 
    #
    # Example:
    #
    # sub-01_run-03_trial-14.wav 
    # ---------------------------------------------------------
    syllable_lookup = dict(
        zip(
            input_df["wav_file"].astype(str),
            input_df["syllable"].astype(str)))

    # ---------------------------------------------------------
    # Process each TSV row / WAV
    # ---------------------------------------------------------
    for row_index, row in input_df.iterrows():

        subject = str(
            row["subject"])

        run = row["run"]
        trial = row["trial"]

        wav_file = str(row["wav_file"])

        # -----------------------------------------------------
        # Get the syllable from INPUT_TSV_PATH based on wav_file
        # -----------------------------------------------------
        syllable = syllable_lookup[
            wav_file]

        wav_path = os.path.join(
            WAV_FOLDER,
            wav_file)

        print("\n" + "-" * 80)
        print(f"[{row_index + 1}/{total_trials}]")
        print(f"Subject  : {subject}")
        print(f"Run      : {run}")
        print(f"Trial    : {trial}")
        print(f"Syllable : {syllable}")
        print(f"WAV      : {wav_file}")

        # -----------------------------------------------------
        # WAV must exist
        # -----------------------------------------------------
        if not os.path.isfile(
            wav_path):

            print("[ERROR] WAV file not found.")

            failed_files.append({
                "subject": subject,
                "run": run,
                "trial": trial,
                "wav_file": wav_file,
                "error": "WAV file not found",})

            continue

        # -----------------------------------------------------
        # Run model
        # -----------------------------------------------------
        try:

            ( onset_times,
                offset_times) = detect_onsets_offsets_from_wav(
                WAV_PATH=wav_path,
                model=model,
                feature_extractor=feature_extractor,
                device=device,)

        except Exception as error:

            print(f"[ERROR] {error}")

            failed_files.append({
                "subject": subject,
                "run": run,
                "trial": trial,
                "wav_file": wav_file,
                "error": str(error),})

            continue

        # -----------------------------------------------------
        # Print detections
        # -----------------------------------------------------
        print("\nDetected onset times:")
        print(onset_times)

        print("\nDetected offset times:")
        print(offset_times)

        # -----------------------------------------------------
        # Safety check
        # -----------------------------------------------------
        if (len(onset_times)!= len(offset_times)):

            print(
                "[ERROR] Different number of "
                "onsets and offsets.")

            failed_files.append({
                "subject": subject,
                "run": run,
                "trial": trial,
                "wav_file": wav_file,
                "error": (
                    f"{len(onset_times)} onsets, "
                    f"{len(offset_times)} offsets"),})
            continue

        number_of_segments = len(
            onset_times)

        print(
            f"\nDetected segments: "
            f"{number_of_segments}")

        # -----------------------------------------------------
        # Create one output TSV row per segment
        # -----------------------------------------------------
        for segment_index, (
            onset,
            offset) in enumerate(
            zip(
                onset_times,
                offset_times),
            start=1):

            output_rows.append({
                "subject": subject,
                "run": run,
                "trial": trial,
                "syllable": syllable,
                "wav_file": wav_file,
                "syllable_number": segment_index,
                "onset_sec": float(onset),
                "offset_sec": float(offset),})

    # =========================================================
    # CREATE OUTPUT DATAFRAME
    # =========================================================

    output_columns = [
        "subject",
        "run",
        "trial",
        "syllable",
        "wav_file",
        "syllable_number",
        "onset_sec",
        "offset_sec",]

    output_df = pd.DataFrame(
        output_rows,
        columns=output_columns)

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------
    if len(output_df) > 0:

        output_df = (
            output_df
            .sort_values(
                by=[
                    "run",
                    "trial",
                    "syllable_number",])
            .reset_index(
                drop=True))

    # ---------------------------------------------------------
    # Output filename
    #
    # sub-01 -> sub_01.tsv
    # sub-02 -> sub_02.tsv
    # ---------------------------------------------------------
    output_subject_name = (
        subject_name.replace("-", "_"))

    output_tsv_path = os.path.join(
        OUTPUT_FOLDER,f"{output_subject_name}.tsv")

    # ---------------------------------------------------------
    # Save TSV
    # ---------------------------------------------------------
    output_df.to_csv(
        output_tsv_path,
        sep="\t",
        index=False,
        float_format="%.4f",)

    # =========================================================
    # FINAL REPORT
    # =========================================================
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Subject: {subject_name}")
    print(f"Input trials: {len(input_df)}")
    print(f"Output syllable segments: " f"{len(output_df)}")
    print(f"Failed trials: " f"{len(failed_files)}")
    print(f"\nOutput TSV saved to:\n" f"{output_tsv_path}")

    # ---------------------------------------------------------
    if failed_files:

        failed_df = pd.DataFrame(
            failed_files)

        failed_path = os.path.join(
            OUTPUT_FOLDER,
            f"{output_subject_name}_failed.tsv")

        failed_df.to_csv(
            failed_path,
            sep="\t",
            index=False)
        print("\nFailed-files report saved to:")
        print(failed_path)
    return output_df



# MAIN
if __name__ == "__main__":
    # Please just give these as inputs and output
    # Folder containing WAV files for ONE subject
    WAV_FOLDER = ("/Users/vahid/Desktop/for_Sivan/Amanda_data/formatted_data/sub-01/")

    # -----------------------------------------------------------------
    # Input TSV for that subject
    INPUT_TSV_PATH = ("/Users/vahid/Desktop/for_Sivan/Amanda_data/Input_TSV/sub-01.tsv")

    # -----------------------------------------------------------------
    # Folder where output TSV will be saved
    OUTPUT_FOLDER = (
        "/Users/vahid/Desktop/for_Sivan/Amanda_data/Output_TSV_testfffff/")
    # -----------------------------------------------------------------
    # Trained checkpoint
    CHECKPOINT = ("./checkpoints/best.pth")

    # Run batch inference for this subject
    output_df = process_subject(
        WAV_FOLDER=WAV_FOLDER,
        INPUT_TSV_PATH=INPUT_TSV_PATH,
        OUTPUT_FOLDER=OUTPUT_FOLDER,
        CHECKPOINT=CHECKPOINT,)






