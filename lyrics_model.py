import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

MAX_LEN = 2000 #每首歌採樣的字數, 超過得不會被訓練到少了會填充0


def load_lyrics_folder(lyrics_folder="lyric", csv_path="id_energy_valence.csv"):
    texts, valence, energy = [], [], []

    # Load the CSV as a DataFrame
    songs = pd.read_csv(csv_path)

    # Make sure the 'id' column is treated as string to match filenames
    songs['id'] = songs['id'].astype(str)

    for _, row in songs.iterrows():
        song_id = row['id']
        lyric_path = os.path.join(lyrics_folder, f"{song_id}.txt")

        if os.path.exists(lyric_path):
            try:
                with open(lyric_path, 'r', encoding='utf-8') as f:
                    lyric = f.read().strip()
                    texts.append(lyric)
                    valence.append(float(row['valence']))
                    energy.append(float(row['energy']))
            except Exception as ex:
                print(f"Error reading {lyric_path}: {ex}")
        else:
            print(f"File {lyric_path} not found.")

    return texts, valence, energy



def load_test_folder(folder_path="test"):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lyric = f.read().strip()
                    results.append((filename, lyric))
            except Exception as ex:
                print(f"Error reading {filename}: {ex}")
    return results


class Vocab:
    def __init__(self, texts):
        self.stoi = {"<pad>": 0, "<unk>": 1}
        idx = 2
        for text in texts:
            for word in re.findall(r"\b\w+\b", text.lower()):
                if word not in self.stoi:
                    self.stoi[word] = idx
                    idx += 1
        self.itos = {i: w for w, i in self.stoi.items()}

    def save_to_txt(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for word, index in self.stoi.items():
                f.write(f"{word}\t{index}\n")

    @staticmethod
    def load_from_txt(filepath):
        stoi = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                word, index = line.strip().split("\t")
                stoi[word] = int(index)
        vocab = Vocab([])
        vocab.stoi = stoi
        vocab.itos = [None] * len(stoi)
        for word, index in stoi.items():
            vocab.itos[index] = word
        return vocab


class LyricsDataset(Dataset):
    def __init__(self, texts, valence, energy, vocab, max_len=MAX_LEN):
        self.texts = [self.tokenize(t, vocab, max_len) for t in texts]
        self.valence = valence
        self.energy = energy

    def tokenize(self, text, vocab, max_len):
        tokens = re.findall(r'\b\w+\b', text.lower())
        indices = [vocab.stoi.get(t, vocab.stoi["<unk>"]) for t in tokens[:max_len]]
        padded = indices + [vocab.stoi["<pad>"]] * (max_len - len(indices))
        return torch.tensor(padded)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], torch.tensor([self.valence[idx], self.energy[idx]], dtype=torch.float)


class LyricsTestDataset(Dataset):
    def __init__(self, texts, vocab, max_len=MAX_LEN):
        self.texts = [self.tokenize(t, vocab, max_len) for t in texts]

    def tokenize(self, text, vocab, max_len):
        tokens = re.findall(r'\b\w+\b', text.lower())
        indices = [vocab.stoi.get(t, vocab.stoi["<unk>"]) for t in tokens[:max_len]]
        padded = indices + [vocab.stoi["<pad>"]] * (max_len - len(indices))
        return torch.tensor(padded)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, output_dim=2, max_len=MAX_LEN, feature_dim=128):
        
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        conv_output_len = max_len // 2 
        self.fc_feature = nn.Sequential(
            nn.Linear(100 * conv_output_len, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, feature_dim),
            nn.ReLU()
        )
        self.fc_output = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)         
        x = x.permute(0, 2, 1)          
        x = self.conv(x)                
        x = x.view(x.size(0), -1) 
        features = self.fc_feature(x)   # 128 feature matrix
        output = self.fc_output(features)    #valence, energy
        return features, output      


def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            feature, outputs = model(inputs)
            # print(feature)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(train_loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_valence_mse, val_energy_mse = evaluate(model, val_loader, loss_fn, device)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Valence MSE: {val_valence_mse:.4f} | Energy MSE: {val_energy_mse:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_lyrics.pt")
            print("Best model saved!")


def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    valence_mse = 0
    energy_mse = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            feature, outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            valence_mse += ((outputs[:, 0] - targets[:, 0]) ** 2).mean().item()
            energy_mse += ((outputs[:, 1] - targets[:, 1]) ** 2).mean().item()
    return total_loss / len(val_loader), valence_mse / len(val_loader), energy_mse / len(val_loader)
