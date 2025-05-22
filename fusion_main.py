from audio_model import AudioCNN
from lyrics_model import Vocab, CNNModel
from fusion_model import FusionModel  
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim

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

#///////////////////////////////////////////////////////////////////////////#
                       # L o a d  D a t a ( train )
#///////////////////////////////////////////////////////////////////////////#
#這部分從 dataset load data，把三種data(audio.mp3 , lyric.txt , labels)放進同一個 data_loader 中，之後把 data_loader 送進 fusion_model

train_ids, train_labels = get_ids_and_labels(csv_path='data/train_labels.csv', Type='train')
train_audios_paths = get_audios_paths(train_ids, audio_file_path='data/audio/')
train_lyrics_paths = get_lyrics_paths(train_ids, lyric_file_path='data/lyrics/')

train_audios = load_audio(train_audios_paths)
train_lyrics = load_lyric(train_lyrics_paths)

vocab_train = Vocab(train_lyrics)

train_lyrics = [tokenize(t, vocab_train, max_len) for t in train_lyrics]  # 把 text tokenize

train_dataset = Combined_Dataset(train_audios, train_lyrics, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize models
audio_model = AudioCNN().to(device)
text_model = CNNModel(len(vocab_train.stoi)).to(device)
fusion_model = FusionModel(audio_dim=128, text_dim=128, output_dim=2).to(device)

#///////////////////////////////////////////////////////////////////////////#
                     # L o a d  D a t a ( validate )
#///////////////////////////////////////////////////////////////////////////#

val_ids, val_labels = get_ids_and_labels(csv_path='data/val_labels.csv', Type='validate')
val_audios_paths = get_audios_paths(val_ids, audio_file_path='data/audio/')
val_lyrics_paths = get_lyrics_paths(val_ids, lyric_file_path='data/lyrics/')

val_audios = load_audio(val_audios_paths)
val_lyrics = load_lyric(val_lyrics_paths)

val_lyrics = [tokenize(t, vocab_train, max_len) for t in val_lyrics]  # 把 text tokenize

val_dataset = Combined_Dataset(val_audios, val_lyrics, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

#///////////////////////////////////////////////////////////////////////////#
                       # L o a d  D a t a ( test )
#///////////////////////////////////////////////////////////////////////////#
#這部分從 dataset load data，把三種data(audio.mp3 , lyric.txt , labels)
#放進同一個 data_loader 中，之後把 data_loader 送進 fusion_model

test_ids, test_labels = get_ids_and_labels(csv_path='data/test_labels.csv', Type='test')
test_audios_paths = get_audios_paths(test_ids, audio_file_path='data/audio/')
test_lyrics_paths = get_lyrics_paths(test_ids, lyric_file_path='data/lyrics/')

test_audios = load_audio(test_audios_paths)
test_lyrics = load_lyric(test_lyrics_paths)

test_lyrics = [tokenize(t, vocab_train, max_len) for t in test_lyrics]  # 把 text tokenize

test_dataset = Combined_Dataset(test_audios, test_lyrics, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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
    audio_model.train()
    text_model.train()
    fusion_model.train()
    total_loss = 0

    for audios, lyrics, labels in train_loader:
        audios, lyrics, labels = audios.to(device), lyrics.to(device), labels.to(device)

        audios_feature = audio_model(audios) #得到 audio feature_vector，到時候作為 input 送給 fusion_model.train()
        lyrics_feature, _ = text_model(lyrics) #得到 lyric feature_vector，到時候作為 input 送給 fusion_model.train()

        predict_value = fusion_model(audios_feature, lyrics_feature)

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

audio_model.eval()
text_model.eval()
fusion_model.eval()

val_losses = []
total_val_loss = 0

with torch.no_grad():
    for audios, lyrics, labels in val_loader:
        audios = audios.to(device)
        lyrics = lyrics.to(device)
        labels = labels.to(device)

        audios_feature_val = audio_model(audios) #得到 audio feature_vector，到時候作為 input 送給 fusion_model.validate()
        lyrics_feature_val, _ = text_model(lyrics) #得到 lyric feature_vector，到時候作為 input 送給 fusion_model.validate()

        predict_value_val = fusion_model(audios_feature_val, lyrics_feature_val)

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

audio_model.eval()
text_model.eval()
fusion_model.eval()

test_losses = []
total_test_loss = 0

with torch.no_grad():
    for audios, lyrics, labels in test_loader:
        audios = audios.to(device)
        lyrics = lyrics.to(device)
        labels = labels.to(device)

        audios_feature_test = audio_model(audios) #得到 audio feature_vector，到時候作為 input 送給 fusion_model.validate()
        lyrics_feature_test, _ = text_model(lyrics) #得到 lyric feature_vector，到時候作為 input 送給 fusion_model.validate()

        predict_value_test = fusion_model(audios_feature_test, lyrics_feature_test)

        test_loss = criterion(predict_value_test, labels)
        test_losses.append(test_loss.item())
        total_test_loss += test_loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f"Epoch: {epoch+1} finished - Test Loss: {avg_test_loss:.4f}")


#///////////////////////////////////////////////////////////////////////////#
                          # p l o t & o u t p u t
#///////////////////////////////////////////////////////////////////////////#
# output 一些圖表跟一些 csv 檔
