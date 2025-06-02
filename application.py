import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import difflib

predictions_df = pd.read_csv("prediction_with_song_names.csv") 

all_names = predictions_df["name"].tolist()
features = predictions_df[["energy", "valence"]].values


def recommend_by_name(song_name):
    song_name = song_name.lower()
    name_map = {name.lower(): name for name in all_names}
    
    best_match = difflib.get_close_matches(song_name, list(name_map.keys()), n=1)
    if not best_match:
        print(" 找不到相似的歌曲名稱。")
        return

    matched_lower = best_match[0]
    matched_name = name_map[matched_lower]

    row = predictions_df[predictions_df["name"].str.lower() == matched_lower].iloc[0]
    query_vector = np.array([[row["energy"], row["valence"]]])

    similarities = cosine_similarity(query_vector, features)[0]
    top_k = similarities.argsort()[::-1][1:6]

    print(f"\n 推薦與《{matched_name}》相似的歌曲：")
    for i in top_k:
        rec_row = predictions_df.iloc[i]
        print(f"{rec_row['name']} | energy={rec_row['energy']:.3f}, valence={rec_row['valence']:.3f}")


def recommend_by_emotion(valence, energy):
    query_vector = np.array([[energy, valence]])
    similarities = cosine_similarity(query_vector, features)[0]
    top_k = similarities.argsort()[::-1][:5]

    print(f"\n 根據情緒 energy={energy:.2f}, valence={valence:.2f} 推薦：")
    for i in top_k:
        rec_row = predictions_df.iloc[i]
        print(f" {rec_row['name']} | energy={rec_row['energy']:.3f}, valence={rec_row['valence']:.3f}")

print(" Welcome to the Music Recommender ")

while True:
    choice = input("\nUse (S)ong name or (E)nergy & Valence? (Q to quit): ").strip().upper()
    if choice == "Q":
        print(" Bye!")
        break
    elif choice == "S":
        song_name = input("Enter song name: ").strip()
        recommend_by_name(song_name)
    elif choice == "E":
        try:
            valence = float(input("Enter valence (0~1): "))
            energy = float(input("Enter energy (0~1): "))
            recommend_by_emotion(valence, energy)
        except ValueError:
            print(" 請輸入正確的數字格式。")
    else:
        print(" Invalid choice.")
