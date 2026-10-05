**We have different scripts in this folder that you can find information regarding each separately here.**
---------------------------------------

## Conver_syllable_column_to_list
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

Intervals with an empty label or the label `silent` are ignored.

### Input

The input is a folder containing `.TextGrid` files.

The expected filename format is:

```text
sub-02_run-08_trial-25.TextGrid
```

Each TextGrid should contain labeled intervals with their corresponding `xmin` and `xmax` values.

### Output

The code combines all TextGrid annotations into **one TSV file** with the following columns:

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

Each non-silent syllable interval is stored as one row. The corresponding WAV filename is also automatically generated from the TextGrid filename.

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

For example, a labeled TextGrid interval containing `pa` from `0.5200` to `0.7100` seconds will produce a TSV row containing the syllable label together with its onset and offset times.

The function returns the generated Pandas DataFrame and also saves it to the specified `OUTPUT_TSV_PATH`.
