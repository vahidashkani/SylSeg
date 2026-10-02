#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 14:40:54 2026

@author: vahid
"""

import io
import os
from contextlib import redirect_stdout
import pandas as pd
import torch
from transformers import AutoFeatureExtractor
from helpers_functions.inference_helpers import (
    create_model_for_test,
    detect_onsets_offsets_from_wav, load_input_tsv,)

# =============================================================================
# SUBJECT-LEVEL BATCH PROCESSING / SINGLE-WAV PROCESSING
def process_subject(WAV_FOLDER: str, INPUT_TSV_PATH: str = None, OUTPUT_FOLDER: str = None, CHECKPOINT: str = None,):
    
    """
    Two modes are supported.

    1. WAV_FOLDER is one WAV file:

       Only run onset/offset detection for that WAV.
       INPUT_TSV_PATH and OUTPUT_FOLDER are not required.

    2. WAV_FOLDER is a directory:

       Process all WAV files listed in the subject TSV.

       List-formatted "syllable" values are used normally, for example:

           ["pa"]
           ["pa", "ta", "ka"]
           [“pa”, “ta”, “ka”]

       If a value is NOT in list format, that WAV is still processed.
       Its detected segments are written to the output TSV, and the
       "syllable" field contains a meaningful list-format error.

       If a valid list contains one label, that label is assigned only
       to the first detected segment.

       If a valid list contains multiple labels, the number of labels
       must match the number of detected segments, and labels are
       assigned to segments in order.

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
    # Checkpoint must exist in both modes
    if (CHECKPOINT is None or not os.path.isfile(CHECKPOINT)):
        raise FileNotFoundError(f"Checkpoint does not exist:\n"
            f"{CHECKPOINT}")

    # =========================================================
    # SINGLE-WAV MODE
    if os.path.isfile(WAV_FOLDER):

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu")

        # -----------------------------------------------------
        # Load feature extractor and model without extra prints
        with redirect_stdout(io.StringIO()):
            feature_extractor = (AutoFeatureExtractor.from_pretrained("openai/whisper-tiny"))
            model = create_model_for_test(device=device, CHECKPOINT=CHECKPOINT)
            
        # -----------------------------------------------------
        # Run model
        # -----------------------------------------------------
        (onset_times, offset_times) = detect_onsets_offsets_from_wav(
            WAV_PATH=WAV_FOLDER,
            model=model,
            feature_extractor=feature_extractor,
            device=device,)

        # -----------------------------------------------------
        # Print only onset and offset lists
        print("Detected onset times:")
        print(onset_times.tolist())

        print("Detected offset times:")
        print(offset_times.tolist())

        return (onset_times, offset_times)

    # =========================================================
    # FOLDER / TSV MODE

    # ---------------------------------------------------------
    # Validate paths
    # ---------------------------------------------------------
    if not os.path.isdir(
        WAV_FOLDER):
        raise NotADirectoryError(
            f"WAV folder or WAV file does not exist:\n"
            f"{WAV_FOLDER}")

    if INPUT_TSV_PATH is None:
        raise ValueError(
            "INPUT_TSV_PATH is required when "
            "WAV_FOLDER is a directory.")

    if OUTPUT_FOLDER is None:

        raise ValueError(
            "OUTPUT_FOLDER is required when "
            "WAV_FOLDER is a directory.")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ---------------------------------------------------------
    # Load TSV
    # load_input_tsv() converts valid list-formatted "syllable"
    # values to Python lists.
    # Invalid syllable-format values are marked with an error
    # but do NOT stop the batch.
    # ---------------------------------------------------------
    input_df = load_input_tsv(
        INPUT_TSV_PATH)

    # ---------------------------------------------------------
    # Determine subject
    subjects = (input_df["subject"].astype(str).unique().tolist())

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
    # Syllables are taken from the "syllable" column, 
    # Valid rows contain a Python list in row["syllable"].    
    # ---------------------------------------------------------
    # Process each TSV row / WAV
    # ---------------------------------------------------------
    for row_index, row in input_df.iterrows():
        subject = str(row["subject"])
        run = row["run"]
        trial = row["trial"]
        wav_file = str(row["wav_file"])

        # -----------------------------------------------------
        # Get syllable information from INPUT_TSV_PATH
        # -----------------------------------------------------
        syllable_list = row["syllable"]
        syllable_error = str(row["_syllable_error"])
        original_syllable = row["_syllable_original"]
        wav_path = os.path.join(WAV_FOLDER, wav_file)
        
        print("\n" + "-" * 80)
        print(f"[{row_index + 1}/{total_trials}]")
        print(f"Subject  : {subject}")
        print(f"Run      : {run}")
        print(f"Trial    : {trial}")
        if syllable_error:
            print(f"Syllable : {original_syllable}")
            print(f"[ERROR] {syllable_error}")

        else:
            print(f"Syllable : {syllable_list}")
        print(f"WAV      : {wav_file}")

        # -----------------------------------------------------
        # WAV must exist
        # -----------------------------------------------------
        if not os.path.isfile(wav_path):
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
            (onset_times,
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
        number_of_segments = len(onset_times)

        # -----------------------------------------------------
        # Invalid syllable format:
        #
        # Keep processing this WAV and write all detected
        # segments to the NORMAL output TSV. The syllable
        # column contains the meaningful format error.
        # -----------------------------------------------------
        if syllable_error:
            failed_files.append({"subject": subject,
                "run": run,
                "trial": trial,
                "wav_file": wav_file,
                "error": syllable_error,})

            for segment_index, (onset, offset) in enumerate(zip(onset_times, offset_times), start=1):
                output_rows.append({
                    "subject": subject,
                    "run": run,
                    "trial": trial,
                    "syllable": syllable_error,
                    "wav_file": wav_file,
                    "syllable_number": segment_index,
                    "onset_sec": float(onset),
                    "offset_sec": float(offset),})
            continue

        # -----------------------------------------------------
        # Valid list:
        # Multiple labels must match the detected segments.
        # One-label lists are the special case requested:
        # only the first segment receives that label.
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Create one output TSV row per detected segment
        # -----------------------------------------------------
        for segment_index, (onset, offset) in enumerate(
            zip(onset_times, offset_times), start=1):
            if len(
                syllable_list) == 1:
                if segment_index == 1:
                    syllable = syllable_list[0]
                else:
                    syllable = ""
            else:
                if segment_index <= len(syllable_list):
                    syllable = syllable_list[segment_index - 1]
                else:
                    syllable = ""

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
    output_columns = ["subject",
        "run",
        "trial",
        "syllable",
        "wav_file",
        "syllable_number",
        "onset_sec",
        "offset_sec",]

    output_df = pd.DataFrame(output_rows, columns=output_columns)

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
                    "syllable_number",]).reset_index(drop=True))

    # ---------------------------------------------------------
    # Output filename
    # ---------------------------------------------------------
    output_subject_name = subject_name
    output_tsv_path = os.path.join(
        OUTPUT_FOLDER,
        f"{output_subject_name}.tsv")

    # ---------------------------------------------------------
    # Save TSV
    # ---------------------------------------------------------
    output_df.to_csv(output_tsv_path, sep="\t", index=False, float_format="%.4f",)
    # =========================================================
    # FINAL REPORT
    # =========================================================
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Subject: {subject_name}")
    print(f"Input trials: {len(input_df)}")

    print(
        f"Output syllable segments: "
        f"{len(output_df)}")

    print(
        f"Failed trials: "
        f"{len(failed_files)}")

    print(
        f"\nOutput TSV saved to:\n"
        f"{output_tsv_path}")
    return output_df


# MAIN
if __name__ == "__main__":

    # -----------------------------------------------------------------
    # WAV input
    #
    # OPTION 1: Folder containing WAV files for ONE subject
    # WAV_FOLDER = (
    #     "/Users/vahid/Desktop/for_Sivan/Amanda_data/formatted_data/sub-01/")
    #
    # OPTION 2: One specific WAV file
    # WAV_FOLDER = ("/Users/vahid/Desktop/for_Sivan/Amanda_data/formatted_data/sub-01/sub-01_run-03_trial-14.wav")
    # -----------------------------------------------------------------
    WAV_FOLDER = ("/Users/vahid/Desktop/for_Sivan/Amanda_data/formatted_data/sub-01/")

    # -----------------------------------------------------------------
    # Input TSV for that subject
    # Required only when WAV_FOLDER is a directory.
    # Ignored when WAV_FOLDER is one WAV file.
    # -----------------------------------------------------------------
    INPUT_TSV_PATH = ("/Users/vahid/Desktop/for_Sivan/Amanda_data/Input_TSV/sub-011.tsv")

    # -----------------------------------------------------------------
    # Folder where output TSV will be saved
    # Required only when WAV_FOLDER is a directory.
    # Ignored when WAV_FOLDER is one WAV file.
    # -----------------------------------------------------------------
    OUTPUT_FOLDER = ("/Users/vahid/Desktop/for_Sivan/Amanda_data/Output_TSV_test11111/")
    # -----------------------------------------------------------------
    # Trained checkpoint
    # -----------------------------------------------------------------
    CHECKPOINT = ("./checkpoints/best.pth")
    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------
    if os.path.isfile(
        WAV_FOLDER):
        process_subject(WAV_FOLDER=WAV_FOLDER, CHECKPOINT=CHECKPOINT,)

    else:
        output_df = process_subject(WAV_FOLDER=WAV_FOLDER, INPUT_TSV_PATH=INPUT_TSV_PATH, OUTPUT_FOLDER=OUTPUT_FOLDER, CHECKPOINT=CHECKPOINT,)
