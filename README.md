# AI-final-project-team69 : Music emotion recognition
This project aims to **predict a song’s emotion**, represented by **energy** and **valence**, using its **lyrics**, **audio**, or a combination of both (fusion model).  
We also include a **application** that recommends songs with similar emotional profiles based on predicted values.

---


## Models
- **Lyrics Model**: CNN over word embeddings.
- **Audio Model**: CNN over mel-spectrogram input.
- **Fusion Model**: Late fusion of 128-dim audio and lyrics features into a final regressor.

## Output
- Predicts **valence** and **energy** in the range [0, 1].
- Generates CSV files for evaluation and recommendation.


## System Architecture
![import torch from torch import nn class FusionModel(nn Module) def __init__(self, audio_dim=128, text_dim=128, output_dim=2) super(FusionModel, self) __init__() self fc1 = nn Linear(audio_dim + te (4)](https://github.com/user-attachments/assets/9fe9a97f-4a8b-4453-acb4-a8b607ed3f41)


## Run Examples

```bash
python lyrics_main.py        # Train lyrics model
python audio_main.py         # Train audio model
python fusion_main.py        # Train fusion model
python application.py        # Run the recommendation system
```
-------------


## File Descriptions

| File/Folder                      | Description |
|----------------------------------|-------------|
| `application.py`                | Song recommender by name or valence/energy input. |
| `audio_main.py`                 | Trains and evaluates the **Audio Model**. |
| `audio_model.py`                | CNN model architecture for processing audio features. |
| `audio_predictions.csv`         | Predicted energy and valence from the audio model. |
| `best_model_audio.pt`           | Saved PyTorch model: best-performing audio model. |
| `best_model_lyrics.pt`          | Saved PyTorch model: best-performing lyrics model. |
| `dataset.py`                    | Dataset utilities and audio data loading class. |
| `fusion_main.py`                | Trains and evaluates the **Fusion Model**. |
| `fusion_model.py`               | Late fusion architecture combining audio + lyrics features. |
| `fusion_predictions.csv`        | Fusion model predictions of valence and energy. |
| `lyrics_main.py`                | Trains and evaluates the **Lyrics Model**. |
| `lyrics_model.py`               | CNN model for lyrics input based on tokenized sequences. |
| `lyrics_validation_loss.png`    | Lyrics model loss curve during training. |
| `make_mel.py`                   | Converts `.mp3` files into mel-spectrogram tensors using Librosa. |
| `mergeList.py`                  | Matches prediction outputs with corresponding song names. |
| `prediction_with_song_names.csv`| Predictions (valence/energy) with song names attached. For running the song recommender |
| `README.md`                     | This file. Project description and file documentation. |
| `vocab.txt`                     | Vocabulary dictionary for lyrics tokenization. |

---

