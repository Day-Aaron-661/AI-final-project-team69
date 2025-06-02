from audio_model import AudioCNN
from lyrics_model import Vocab, CNNModel
from fusion_model import FusionModel  
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
from make_mel import process_all_mp3_to_pt
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm

#///////////////////////////////////////////////////////////////////////////#
                          # I n i t i a l i z e
#///////////////////////////////////////////////////////////////////////////#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_len = 2000
batch_size = 16  # Set your desired batch size
vocab = Vocab.load_from_txt("vocab.txt")
# Initialize models
audio_model = AudioCNN().to(device)
text_model = CNNModel(len(vocab.stoi)).to(device)

audio_model.load_state_dict(torch.load('best_model_audio.pt'))
text_model.load_state_dict(torch.load('best_model_lyrics.pt'))

fusion_model = FusionModel(audio_dim=128, text_dim=128, output_dim=2).to(device)
#process_all_mp3_to_pt( audio_dir='data//audio' )

#///////////////////////////////////////////////////////////////////////////#
                       # L o a d  D a t a ( train )
#///////////////////////////////////////////////////////////////////////////#
#這部分從 dataset load data，把三種data(audio.mp3 , lyric.txt , labels)放進同一個 data_loader 中，之後把 data_loader 送進 fusion_model
print("Loading training data...")
train_ids, train_labels = get_ids_and_labels(csv_path='data//labels_4000.csv', Type='train')

train_audios_paths = get_audios_paths(train_ids, audio_file_path='data//audio')
train_lyrics_paths = get_lyrics_paths(train_ids, lyric_file_path='data//lyrics')

train_audios = load_audio(train_audios_paths)
train_lyrics = load_lyric(train_lyrics_paths)

train_lyrics = [tokenize(t, vocab, max_len) for t in train_lyrics]  # 把 text tokenize

train_dataset = Combined_Dataset(train_audios, train_lyrics, train_labels , train_ids)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


#///////////////////////////////////////////////////////////////////////////#
                     # L o a d  D a t a ( validate )
#///////////////////////////////////////////////////////////////////////////#
print("Loading validation data...")
val_ids, val_labels = get_ids_and_labels(csv_path='data//labels_4000.csv', Type='validate')
val_audios_paths = get_audios_paths(val_ids, audio_file_path='data//audio')
val_lyrics_paths = get_lyrics_paths(val_ids, lyric_file_path='data//lyrics')

val_audios = load_audio(val_audios_paths)
val_lyrics = load_lyric(val_lyrics_paths)

val_lyrics = [tokenize(t, vocab, max_len) for t in val_lyrics]  # 把 text tokenize

val_dataset = Combined_Dataset(val_audios, val_lyrics, val_labels , val_ids)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

#///////////////////////////////////////////////////////////////////////////#
                       # L o a d  D a t a ( test )
#///////////////////////////////////////////////////////////////////////////#
#這部分從 dataset load data，把三種data(audio.mp3 , lyric.txt , labels)
#放進同一個 data_loader 中，之後把 data_loader 送進 fusion_model
print("Loading test data...")
test_ids, test_labels = get_ids_and_labels(csv_path='data//labels_4000.csv', Type='test')
test_audios_paths = get_audios_paths(test_ids, audio_file_path='data//audio')
test_lyrics_paths = get_lyrics_paths(test_ids, lyric_file_path='data//lyrics')

test_audios = load_audio(test_audios_paths)
test_lyrics = load_lyric(test_lyrics_paths)

test_lyrics = [tokenize(t, vocab, max_len) for t in test_lyrics]  # 把 text tokenize

test_dataset = Combined_Dataset(test_audios, test_lyrics, test_labels ,test_ids)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#///////////////////////////////////////////////////////////////////////////#
                            # t r a i n i n g
#///////////////////////////////////////////////////////////////////////////#
# 這區域對 fusion_model 做 training，到時候把 for 迴圈留著然後在 fusion_model.py 做 train function

criterion = nn.MSELoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=1e-5 , weight_decay=1e-4)

print("training session")

train_losses = []
val_losses = []
y_true_energy = []
y_pred_energy = []

y_true_valence = []
y_pred_valence = []


EPOCHS = 30
for epoch in range(EPOCHS):
    audio_model.eval()
    text_model.eval()
    fusion_model.train()
    total_train_loss = 0
    total_val_loss = 0

    progress = tqdm( train_loader , desc= 'Training', leave= False)
    for audios, lyrics, labels , id in progress:
        audios, lyrics, labels = audios.to(device), lyrics.to(device), labels.to(device)

        audios_feature, _ = audio_model(audios) #得到 audio feature_vector，到時候作為 input 送給 fusion_model.train()
        lyrics_feature, _ = text_model(lyrics) #得到 lyric feature_vector，到時候作為 input 送給 fusion_model.train()

        predict_value = fusion_model(audios_feature, lyrics_feature)

        train_loss = criterion(predict_value, labels)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_train_loss += train_loss.item()

    with torch.no_grad():
        for audios, lyrics, labels , id in val_loader:
            audios = audios.to(device)
            lyrics = lyrics.to(device)
            labels = labels.to(device)

            audios_feature_val, _ = audio_model(audios) #得到 audio feature_vector，到時候作為 input 送給 fusion_model.validate()
            lyrics_feature_val, _ = text_model(lyrics) #得到 lyric feature_vector，到時候作為 input 送給 fusion_model.validate()

            predict_value_val = fusion_model(audios_feature_val, lyrics_feature_val)

            val_loss = criterion(predict_value_val, labels)
        
            total_val_loss += val_loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch: {epoch+1} finished - Training Loss: {avg_train_loss:.4f} , Val Loss: {avg_val_loss:.4f}")



#///////////////////////////////////////////////////////////////////////////#
                              # t e s t i n g 
#///////////////////////////////////////////////////////////////////////////#
# 這區域對 fusion_model 做 testing，到時候要搬進 fusion_model.py

print("test session")

audio_model.eval()
text_model.eval()
fusion_model.eval()

results = []

with torch.no_grad():
    for audios, lyrics, labels , audio_ids in test_loader:
        audios = audios.to(device)
        lyrics = lyrics.to(device)
        labels = labels.to(device)

        audios_feature_test, _ = audio_model(audios) #得到 audio feature_vector，到時候作為 input 送給 fusion_model.validate()
        lyrics_feature_test, _ = text_model(lyrics) #得到 lyric feature_vector，到時候作為 input 送給 fusion_model.validate()

        predict_value_test = fusion_model(audios_feature_test, lyrics_feature_test)

        pred_energy = predict_value_test[:, 0].detach().cpu().numpy()
        pred_valence = predict_value_test[:, 1].detach().cpu().numpy()

        true_energy = labels[:, 0].cpu().numpy()
        true_valence = labels[:, 1].cpu().numpy()

        audio_ids = audio_ids.tolist()

        for audio_id, energy, valence in zip(audio_ids, pred_energy, pred_valence):
            results.append([audio_id, energy, valence])

        y_true_energy.extend(true_energy)
        y_pred_energy.extend(pred_energy)

        y_true_valence.extend(true_valence)
        y_pred_valence.extend(pred_valence)

    # Calculate R² scores
    
    r2_energy = r2_score(y_true_energy, y_pred_energy)
    r2_valence = r2_score(y_true_valence, y_pred_valence)

    print(f"R² (Energy): {r2_energy:.4f}")
    print(f"R² (Valence): {r2_valence:.4f}")

    test_data = pd.DataFrame( results , columns=['id' , 'energy' , 'valence'])
    test_data.to_csv('fusion_predictions.csv' , index=False)

    print(f"Predictions saved to 'fusion_predictions.csv'")


#///////////////////////////////////////////////////////////////////////////#
                          # p l o t & o u t p u t
#///////////////////////////////////////////////////////////////////////////#
# output 一些圖表跟一些 csv 檔
epochs = list(range(1, len(train_losses) + 1))

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Train Loss", marker='o' , color='blue')
plt.plot(epochs, val_losses, label="Validation Loss", marker='o' , color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()