#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:16:28 2026

@author: vahid
"""

import pandas as pd
import re
import json


def convert_syllable_to_list(value):
    """
    Convert a syllable value to a list. So, please pay attention that
    this code covers the data that are in these below format. I think all of our data are
    in these format.
    
    Examples:
        31452       -> ["3", "1", "4", "5", "2"]
        pa ta ka    -> ["pa", "ta", "ka"]
        pa-ta-ka    -> ["pa", "ta", "ka"]
        pa_ta_ka    -> ["pa", "ta", "ka"]
        day         -> ["day"]
    """

    value = str(value).strip()
    # If the value contains only numbers, split each digit into a separate item.
    if value.isdigit():
        return list(value)

    # Otherwise, split by: space, hyphen (-), or underscore (_)
    items = re.split(r"[\s_-]+", value)
    # Remove empty items
    return [item for item in items if item]


def convert_tsv_syllables(input_tsv, output_tsv):
    """
    Read a TSV file, convert the 'syllable' column to list format,
    and save the result as a new TSV file.
    """

    # Read TSV as strings so values such as 31452 stay unchanged
    df = pd.read_csv(input_tsv, sep="\t", dtype=str)
    if "syllable" not in df.columns:
        raise ValueError("The input TSV does not contain a 'syllable' column.")

    # Convert each syllable value to a list and store it as JSON text
    df["syllable"] = df["syllable"].apply(
        lambda x: json.dumps(convert_syllable_to_list(x), ensure_ascii=False, separators=(",", ":")))

    # Save as TSV
    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"Converted TSV saved to: {output_tsv}")
    
    
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# you can use this function like below to convert your .tsv files into List based TSV file

'''
input_tsv = "/Users/vahid/Desktop/for_Sivan/Amanda_data/Input_TSV/sub-01.tsv"
output_tsv = "/Users/vahid/Desktop/for_Sivan/Amanda_data/Input_TSV/sub-01_converted.tsv"

convert_tsv_syllables(input_tsv, output_tsv)
'''







    