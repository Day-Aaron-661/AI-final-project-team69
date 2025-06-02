import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


pred_df = pd.read_csv("fusion_predictions.csv") 
label_df = pd.read_csv("data\labels.csv")             
song_df = pd.read_csv("song.csv")  

song_features = song_df[["energy", "valence"]].values
song_names = song_df["name"].tolist()

pred_features = pred_df[["energy", "valence"]].values

matched_names = []

for vec in pred_features:
    sims = cosine_similarity([vec], song_features)[0]
    best_match_idx = np.argmax(sims)
    matched_name = song_names[best_match_idx]
    matched_names.append(matched_name)

pred_df["name"] = matched_names

pred_df.to_csv("prediction_with_song_names.csv", index=False)
print("Saved as prediction_with_song_names.csv")
