# SylSeg
---------

**Inference**
to Use the model to produce segmentation borders, you need to run the `inference` code from this repository.
The `inference` script processes the `WAV` files listed in one subject’s input `TSV`. It detects onset and offset times for each WAV and saves one output row per detected segment.

**Note**
Before running the inference code make sure about these:
* Download the last version of the trained weights of the model and put inside the `checkpoints` folder. for mor info refere to `checkpoints`.
* wave files are named in this format `sub-01_run-01_trial-01.wav`
* your input `.tsv` file is in this format. Your `Syllable` column in the input `.tsv` must be in the `List` format.
  <img width="575" height="306" alt="Screenshot 2026-10-02 at 3 05 43 PM" src="https://github.com/user-attachments/assets/390064fa-b621-4bc0-be44-b0d599080555" />



# Before running
-----------------
Set these four paths in the **__main__** section inside the `inference` script:
* WAV_FOLDER = "/path/to/subject/wav/files"
* INPUT_TSV_PATH = "/path/to/input_tsv/sub-01.tsv"
* OUTPUT_FOLDER = "/path/to/output/folder"
* CHECKPOINT = "/path/to/checkpoints/best.pth"






