import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from make_mel import mp3_to_mel
from torch.utils.data import Dataset

class Combined_Dataset(Dataset):
    def __init__(self, audios , lyrics, labels , ids):
        self.audios = audios
        self.lyrics = lyrics
        self.labels = labels  
        self.ids = ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audios[idx], self.lyrics[idx], self.labels[idx] , self.ids[idx]

class Audio_Dataset(Dataset):
    def __init__(self, audios ,labels , ids):
        self.audios = audios
        self.labels = labels  
        self.ids = ids

    def __len__(self):

        return len(self.ids)

    def __getitem__(self, idx):
        return self.audios[idx], self.labels[idx] , self.ids[idx]
    
def load_audio( audio_paths ):

    audios = []

    for audio_path in audio_paths:
        if audio_path.endswith('.pt'):
            audio = torch.load(audio_path)
        elif audio_path.endswith('.mp3'):
            audio = mp3_to_mel(audio_path)
        audios.append(audio)
    # print(len(audios))
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
    data = data[data['type'] == Type].reset_index(drop=True)
    data = data.sort_values("id").reset_index(drop=True)

    ids = data['id']
    labels = torch.tensor(data[['energy', 'valence']].values, dtype=torch.float32)
    #print(len(labels))
    return ids , labels

def get_audios_paths ( ids , audio_file_path ):

    audio_paths = []
    mel_tensor_path = 'mel_tensors'
    for song_id in ids:
        pt_path = os.path.join(mel_tensor_path, f"{song_id}.pt")
        mp3_path = os.path.join(audio_file_path, f"{song_id}.mp3")

        if os.path.exists(pt_path):
            audio_paths.append(pt_path)
        elif os.path.exists(mp3_path):
            audio_paths.append(mp3_path)
    return audio_paths

def get_lyrics_paths ( ids , lyric_file_path ):

    lyric_paths = []

    for song_id in ids:
        lyric_path = os.path.join(lyric_file_path, f"{song_id}.txt")
        lyric_paths.append( lyric_path )
    
    return lyric_paths

def tokenize ( text , vocab , max_len = 2000 ):
    tokens = re.findall(r'\b\w+\b', text.lower())
    indices = [vocab.stoi.get(t, vocab.stoi["<unk>"]) for t in tokens[:max_len]]
    padded = indices + [vocab.stoi["<pad>"]] * (max_len - len(indices))
    return torch.tensor(padded)
