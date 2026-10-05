### Inference Utility Functions

This code contains the supporting functions used by the main `inference` code for speech onset and offset detection.

It includes functions for **audio loading, feature extraction, model and checkpoint loading, onset/offset detection and post-processing, frame-to-time conversion, and input TSV validation**.  

These functions are placed in a separate file to keep the main `inference` code clean, simple, and easy to follow. The `inference` code imports and uses these functions when processing the input WAV files and generating the predicted segmentation boundaries.

The code also validates the input TSV file and ensures that the `syllable` column follows the required list format before processing.
