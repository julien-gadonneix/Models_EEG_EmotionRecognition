# Emotion Detection for Enhanced Psychiatric Diagnosis

## Research internship in the Biosignall processing lab at DTU

### Short description

Most of the code is written in Python and uses the PyTorch library for Deep Learning.

Emotional states play a crucial role in psychological health, influencing conditions like depression and presenting
challenges due to their subjective nature. Recent advances in emotion detection systems offered promising glimpses for
improving psychiatric assessments. Specifically, Brain-Computer Interface technologies provide a direct interface between brain
activity and computational systems. This enables a possibly very accurate classification of emotional states using EEG signals.

This project explores forward-looking deep learning techniques such as:

- the deptwise separable convolutional layer,
- the Capsule network or
- transformer-based models like the Swin transformer.

They are discussed for their effectiveness in extracting temporal, spatial, and frequency features from EEG data that are
processed with signal processing techniques like:

- filtering,
- continuous wavelet transforms or
- empirical mode decomposition.

Moreover, the report focuses on self-supervised learning and contrastive learning for emotion classification across diverse patient
populations. The potential of these frameworks to provide personalized insights is highlighted making a connection between technologies
and mental health diagnostics.

### Key files:

- **`models/`**: Implementation of all the DL models.
- **`preprocess/`**: Adaptative preprocessing of the EEG datasets.
- **`eval_[dataset].py`**: Python script for evaluating the models and computing the metrics over several runs via the ray library.
- **`[dataset].py`**: Python script for optimizing the hyperparameters of the models and processes via the ray library.
- **`tools.py`**: Helper functions and constants.
