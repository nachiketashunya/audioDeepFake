# Audio DeepFake Detection

This repository provides tools and models for detecting audio deepfakes, which are synthetically generated speech designed to mimic a specific person's voice.

**Implemented Models:**

* **ShallowCNN:** A lightweight Convolutional Neural Network (CNN) model suitable for real-time inference on resource-constrained devices.
* **SimpleLSTM:** A Long Short-Term Memory (LSTM) network that effectively captures long-range dependencies within audio sequences.
* **DTDNN (Densely Connected Time Delay Neural Network):** A powerful architecture that leverages time delays and dense connections to achieve high detection accuracy.

**Training Datasets:**

The models are trained on a combination of:

* **LJ Speech Dataset:** A large, high-quality corpus of clean speech recordings.
* **WaveFake Dataset:** A collection of audio deepfakes created using various deepfake generation techniques. 

**Model Download:**

Pre-trained models for each architecture are available for download at the following location:

[link_to_download_models]

**Audio Inference:**

To analyze an audio file and determine its likelihood of being a deepfake, run the following command in your terminal:

```bash
python main.py audio_path
```


audioDeepFake
==============================

This project is for Audio Deep Fake detection.

Project Organization
------------

    ├── LICENSE
    ├── README.md                               <- The top-level README for developers using this project.
    ├── data
    │   ├── processed                           <- The final, canonical data sets for modeling.
    │   └── raw                                 <- The original, immutable data dump.
    │
    ├── models                                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                               <- Jupyter notebooks. 
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment, e.g.
    │                                              generated with `pip freeze > requirements.txt`
    │
    ├── audioDeepFake           <- Source code for use in this project.
    │   │
    │   ├── data                                <- Scripts to download or generate data
    │   │
    │   ├── utils                                <- Scripts utilities used during data generation or training
    │   │
    │   ├── training                            <- Scripts to train models
    │   │
    │   ├── inference                            <- Scripts to inference the models
    │   │
    │   └── models                              <- Scripts to define the models
