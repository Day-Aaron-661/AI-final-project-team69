import os
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle

from lyrics_model import (
    load_lyrics_folder, load_test_folder,
    Vocab, LyricsDataset, LyricsTestDataset, CNNModel, train
)

# Load training and validation data
train_texts, train_valence, train_energy = load_lyrics_folder("lyric", "id_energy_valence.csv")
val_texts, val_valence, val_energy = load_lyrics_folder("lyric", "id_energy_valence.csv")

# Build and save vocab
vocab = Vocab(train_texts)

vocab.save_to_txt("vocab.txt")


# Prepare datasets and dataloaders
train_dataset = LyricsDataset(train_texts, train_valence, train_energy, vocab)
val_dataset = LyricsDataset(val_texts, val_valence, val_energy, vocab)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(len(vocab.stoi)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Train the model
train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=30)


# Load vocab and model
vocab = Vocab.load_from_txt("vocab.txt")

model = CNNModel(len(vocab.stoi)).to(device)
model.load_state_dict(torch.load("best_model_lyrics.pt"))
model.eval()

# Load test lyrics

test_data = load_test_folder("lyric")
test_texts = [lyric for _, lyric in sorted(test_data, key=lambda x: x[0])]
filenames = [filename for filename, _ in sorted(test_data, key=lambda x: x[0])]

test_dataset = LyricsTestDataset(test_texts, vocab)
results = []

for i in range(len(test_dataset)):
    token = test_dataset[i].unsqueeze(0).to(device)
    with torch.no_grad():
        feature, output = model(token)
        valence = output[0][0].item()
        energy = output[0][1].item()
    results.append([filenames[i], valence, energy])

# Write predictions to CSV
with open("test_predictions.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "valence", "energy"])
    writer.writerows(results)

print("Prediction saved to test_predictions.csv")
