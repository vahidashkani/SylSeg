**We have different scripts in this folder that you can find information regarding each separately here.**
---------------------------------------

## Convert_syllable_column_to_list
------------------------------------

The inference code requires the syllable column in it's input `.tsv` file to be in `list` format. If your TSV file does not already follow this format, and you may have syllable in formats such as:

```text
31452       
pa ta ka    
pa-ta-ka    
pa_ta_ka    
day         
```
So, you can use this function to convert it to **list format required by the inference code**. Then, the converted TSV file can be like:

```text
31452       → ["3","1","4","5","2"]
pa ta ka    → ["pa","ta","ka"]
pa-ta-ka    → ["pa","ta","ka"]
pa_ta_ka    → ["pa","ta","ka"]
day         → ["day"]
```

### How to Use

Set the path of your original TSV file and the path where you want to save the converted file:
```python
input_tsv = "/path/to/sub-01.tsv"
output_tsv = "/path/to/sub-01_converted.tsv"

convert_tsv_syllables(input_tsv, output_tsv)
```

The converted file will be saved to the specified `output_tsv` path and can then be used as input to the inference code.



## TextGrids_to_tsv
----------------------------

This code converts **Praat TextGrid annotation files into a single TSV file** that can be used for training, evaluation, or analysis.

The code reads all `.TextGrid` files from a specified folder, extracts the **non-silent syllable intervals**, and saves their labels, onset times, and offset times in a structured `.tsv` file.

### Input

The input is a folder containing `.TextGrid` files.

The expected filename format is:

```text
sub-02_run-08_trial-25.TextGrid
```

### Output

The code combines all TextGrid annotations into **one TSV file** with the following columns. 

```text
subject
run
trial
syllable
wav_file
syllable_number
onset_sec
offset_sec
```

### How to Use

Set the folder containing the TextGrid files and the desired output TSV path:

```python
TEXTGRID_FOLDER = "/path/to/TextGrid/folder/"
OUTPUT_TSV_PATH = "/path/to/output/sub_01.tsv"

textgrids_to_output_tsv(
    textgrid_folder=TEXTGRID_FOLDER,
    output_tsv_path=OUTPUT_TSV_PATH,
    silence_label="silent",
)
```

## tsv_to_TextGrid
-------------------------

This code converts the **segmentation results stored in an output TSV file into Praat TextGrid files**. One separate TextGrid is generated for each WAV file/trial.

The detected onset and offset times are used to create the syllable intervals. The gaps between syllables are automatically labeled as `silent`, and the TextGrid covers the complete duration of the corresponding WAV file.

### Input

The code requires:

1. An **output TSV file** containing the segmentation results with the following columns:

```text
subject
run
trial
syllable
wav_file
syllable_number
onset_sec
offset_sec
```

2. A **WAV folder** containing the original WAV files. The WAV files are used to obtain the exact duration of each recording.

### Output

The output is a folder containing **one `.TextGrid` file for each WAV file/trial**.


### How to Use

Set the following paths:

```python
INPUT_TSV_PATH = "/path/to/output/sub_01.tsv"

WAV_FOLDER = "/path/to/wav/sub-01/"

OUTPUT_TEXTGRID_FOLDER = "/path/to/TextGrids/sub-01/"
```

Then run:

```python
output_tsv_to_textgrids(
    input_tsv_path=INPUT_TSV_PATH,
    wav_folder=WAV_FOLDER,
    output_folder=OUTPUT_TEXTGRID_FOLDER,
    use_syllable_label=True,
)
```

Set `use_syllable_label=True` to include the syllable labels from the TSV file in the generated TextGrids.

If any WAV file cannot be converted, the code records the error in `failed_textgrid_conversion.tsv` inside the output folder.
