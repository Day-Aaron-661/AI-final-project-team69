import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from CNN import mp3_to_mel
from torch.utils.data import Dataset

class Combined_Dataset(Dataset):
    def __init__(self, audios , lyrics, labels ):
        self.audios = audios
        self.lyrics = lyrics
        self.labels = labels  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audios[idx], self.lyrics[idx], self.labels[idx]

    
def load_audio( audio_paths ):

    audios = []

    for audio_path in audio_paths:

        if os.path.exists(audio_path):
            audio = mp3_to_mel(audio_path)
            audios.append(audio)
    
    return audios

def load_lyric( lyric_paths ):

    lyrics = []

    for lyric_path in lyric_paths:
        if os.path.exists(lyric_path):
            try:
                with open(lyric_path, 'r', encoding='utf-8') as f:
                    lyric = f.read().strip()
                    lyrics.append(lyric)
            except Exception as ex:
                    print(f"Error reading {lyric_path}: {ex}")
        else:
            print(f"File {lyric_path} not found.")
    
    return lyrics


def get_ids_and_labels( csv_path , Type ):

    data = pd.read_csv(csv_path)
    data = data[data['Type'] == Type].reset_index(drop=True)
    data = data.sort_values("id").reset_index(drop=True)

    ids = data['id']
    labels = data[['energy','valence']].to_numpy()
    
    return ids , labels

def get_audios_paths ( ids , audio_file_path ):

    audio_paths = []

    for song_id in ids:
        audio_path = os.path.join(audio_file_path, f"{song_id}.mp3")
        audio_paths.append( audio_path )
    
    return audio_paths

def get_lyrics_paths ( ids , lyric_file_path ):

    lyric_paths = []

    for song_id in ids:
        lyric_path = os.path.join(lyric_file_path, f"{song_id}.mp3")
        lyric_paths.append( lyric_path )
    
    return lyric_paths