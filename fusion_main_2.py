from lyrics_model import Vocab
from fusion_model_2 import LateFusionModel  
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import pandas as pd
import dataset
from dataset import (
    Combined_Dataset, tokenize, get_audios_paths,
    get_ids_and_labels, get_lyrics_paths, load_audio,
    load_lyric
)


#///////////////////////////////////////////////////////////////////////////#
                          # I n i t i a l i z e
#///////////////////////////////////////////////////////////////////////////#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_len = 2000
batch_size = 16  # Set your desired batch size
vocab = Vocab.load_from_txt("vocab.txt")
# Initialize models

fusion_model = LateFusionModel( vocab_size=len(vocab.stoi) , audio_dim=128, text_dim=128 , max_len=max_len , output_dim=2).to(device)

#///////////////////////////////////////////////////////////////////////////#
                       # L o a d  D a t a ( train )
#///////////////////////////////////////////////////////////////////////////#
#這部分從 dataset load data，把三種data(audio.mp3 , lyric.txt , labels)放進同一個 data_loader 中，之後把 data_loader 送進 fusion_model

train_ids, train_labels = get_ids_and_labels(csv_path='data//labels.csv', Type='train')

train_audios_paths = get_audios_paths(train_ids, audio_file_path='data//audio')
train_lyrics_paths = get_lyrics_paths(train_ids, lyric_file_path='data//lyrics')

train_audios = load_audio(train_audios_paths)
train_lyrics = load_lyric(train_lyrics_paths)

train_lyrics = [tokenize(t, vocab, max_len) for t in train_lyrics]  # 把 text tokenize

train_dataset = Combined_Dataset(train_audios, train_lyrics, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


#///////////////////////////////////////////////////////////////////////////#
                     # L o a d  D a t a ( validate )
#///////////////////////////////////////////////////////////////////////////#

val_ids, val_labels = get_ids_and_labels(csv_path='data//labels.csv', Type='validate')
val_audios_paths = get_audios_paths(val_ids, audio_file_path='data//audio')
val_lyrics_paths = get_lyrics_paths(val_ids, lyric_file_path='data//lyrics')

val_audios = load_audio(val_audios_paths)
val_lyrics = load_lyric(val_lyrics_paths)

val_lyrics = [tokenize(t, vocab, max_len) for t in val_lyrics]  # 把 text tokenize

val_dataset = Combined_Dataset(val_audios, val_lyrics, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

#///////////////////////////////////////////////////////////////////////////#
                       # L o a d  D a t a ( test )
#///////////////////////////////////////////////////////////////////////////#
#這部分從 dataset load data，把三種data(audio.mp3 , lyric.txt , labels)
#放進同一個 data_loader 中，之後把 data_loader 送進 fusion_model

test_ids, test_labels = get_ids_and_labels(csv_path='data//labels.csv', Type='train')
test_audios_paths = get_audios_paths(test_ids, audio_file_path='data//audio')
test_lyrics_paths = get_lyrics_paths(test_ids, lyric_file_path='data//lyrics')

test_audios = load_audio(test_audios_paths)
test_lyrics = load_lyric(test_lyrics_paths)

test_lyrics = [tokenize(t, vocab, max_len) for t in test_lyrics]  # 把 text tokenize

test_dataset = Combined_Dataset(test_audios, test_lyrics, test_ids)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#///////////////////////////////////////////////////////////////////////////#
                            # t r a i n i n g
#///////////////////////////////////////////////////////////////////////////#
# 這區域對 fusion_model 做 training，到時候把 for 迴圈留著然後在 fusion_model.py 做 train function

criterion = nn.MSELoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=1e-4)

print("training session")

train_losses = []
val_losses = []

EPOCHS = 10
for epoch in range(EPOCHS):
    fusion_model.train()
    total_loss = 0

    for audios, lyrics, labels in train_loader:
        audios, lyrics, labels = audios.to(device), lyrics.to(device), labels.to(device)

        predict_value = fusion_model(audios, lyrics)

        train_loss = criterion(predict_value, labels)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_loss += train_loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch: {epoch+1} finished - Training Loss: {avg_loss:.4f}")


#///////////////////////////////////////////////////////////////////////////#
                            # v a l i d a t e 
#///////////////////////////////////////////////////////////////////////////#
# 這區域對 fusion_model 做 validating，到時候把 for 迴圈留著然後在 fusion_model.py 做 validate function

print("validate session")

fusion_model.eval()

val_losses = []
total_val_loss = 0

with torch.no_grad():
    for audios, lyrics, labels in val_loader:
        audios = audios.to(device)
        lyrics = lyrics.to(device)
        labels = labels.to(device)

        predict_value_val = fusion_model(audios, lyrics)

        val_loss = criterion(predict_value_val, labels)
        val_losses.append(val_loss.item())
        total_val_loss += val_loss.item()

avg_val_loss = total_val_loss / len(val_loader)
print(f"Epoch: {epoch+1} finished - Val Loss: {avg_val_loss:.4f}")


#///////////////////////////////////////////////////////////////////////////#
                              # t e s t i n g 
#///////////////////////////////////////////////////////////////////////////#
# 這區域對 fusion_model 做 testing，到時候要搬進 fusion_model.py

print("test session")

fusion_model.eval()

test_losses = []
total_test_loss = 0
results = []

with torch.no_grad():
    for audios, lyrics, audio_ids in test_loader:
        audios = audios.to(device)
        lyrics = lyrics.to(device)
        labels = labels.to(device)

        predict_value_test = fusion_model(audios, lyrics)

        # test_loss = criterion(predict_value_test, labels)
        # test_losses.append(test_loss.item())
        # total_test_loss += test_loss.item()

        energies = predict_value_test[:, 0].tolist() 
        valences = predict_value_test[:, 1].tolist()
        audio_ids = audio_ids.tolist()

        for audio_id, energy, valence in zip(audio_ids, energies, valences):
            results.append([audio_id, energy, valence])

    test_data = pd.DataFrame( results , columns=['id' , 'energy' , 'valence'])
    test_data.to_csv('fusion_predictions_2.csv' , index=False)

    print(f"Predictions saved to 'fusion_predictions_2.csv'")

# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Epoch: {epoch+1} finished - Test Loss: {avg_test_loss:.4f}")


#///////////////////////////////////////////////////////////////////////////#
                          # p l o t & o u t p u t
#///////////////////////////////////////////////////////////////////////////#
# output 一些圖表跟一些 csv 檔
