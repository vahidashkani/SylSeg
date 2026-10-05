**We have different scripts in this folder that you can find information regarding each separately here.**
---------------------------------------

## Conver_syllable_column_to_list
------------------------------------

The inference code requires the syllable column in the input .tsv file to be in list format. If your TSV file does not already follow this format, the syllable information may appear in formats such as:

```text
31452       → ["3","1","4","5","2"]
pa ta ka    → ["pa","ta","ka"]
pa-ta-ka    → ["pa","ta","ka"]
pa_ta_ka    → ["pa","ta","ka"]
day         → ["day"]
```
So, you can use this function to convert it to **list format required by the inference code**. The converted TSV file keeps the original columns and only reformats the `syllable` column.

### How to Use

Set the path of your original TSV file and the path where you want to save the converted file:
```python
input_tsv = "/path/to/sub-01.tsv"
output_tsv = "/path/to/sub-01_converted.tsv"

convert_tsv_syllables(input_tsv, output_tsv)
```

The converted file will be saved to the specified `output_tsv` path and can then be used as input to the inference code.



## convert_TextGrids_to_tsv
----------------------------
