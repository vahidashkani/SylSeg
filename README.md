# SylSeg
---------

**Inference**

To generate segmentation borders using the model, run the `inference` code in this repository.

The inference script processes the `WAV` files listed in one subject’s input `TSV`. It detects the onset and offset times in each WAV file and saves one output row for each detected segment.

**Note#1**  

Before running the inference code, make sure that:

* You have downloaded the latest trained model weights and placed them in the `checkpoints` folder. For more information, see the `checkpoints` folder.
* Your WAV files follow this naming format: `sub-01_run-01_trial-01.wav`.
* Your input `.tsv` file follows the required format. The `Syllable` column must contain values in list format.
  
  <img width="575" height="306" alt="Screenshot 2026-10-02 at 3 05 43 PM" src="https://github.com/user-attachments/assets/390064fa-b621-4bc0-be44-b0d599080555" />

**Note#2**

If you set the full path to a wave file like 

WAV_FOLDER = ("/path/to/subject/wav/sub-01_run-03_trial-14.wav")

then the code will return list of `onset` and `offset` borders for that wave file.

# Before running
-----------------
Set these four paths in the `__main__` section inside the `inference` script:

* WAV_FOLDER = "/path/to/subject/wav/files"
* INPUT_TSV_PATH = "/path/to/input_tsv/sub-01.tsv"
* OUTPUT_FOLDER = "/path/to/output/folder"
* CHECKPOINT = "/path/to/checkpoints/best.pth"






