### GRBASPredictor

The `model` is a PyTorch-based neural network designed to detect **speech onset and offset boundaries** at the frame level.

The model extracts hidden representations from the last three encoder layers of a pretrained `ASR` model. These representations are projected through learnable adapters and combined using trainable layer weights. The resulting features are then processed by a local-context `CNN` followed by a `unidirectional LSTM` to capture temporal information.

Finally, two independent linear prediction heads produce:

- **Onset logits** – indicate the likelihood of a speech onset at each frame.
- **Offset logits** – indicate the likelihood of a speech offset at each frame.

The model therefore converts acoustic representations into frame-level onset and offset predictions that can be used for speech/syllable segmentation.
