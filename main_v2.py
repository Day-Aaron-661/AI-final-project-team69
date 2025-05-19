from CNN import AudioCNN
#from BERT import TextEncoder
#from Fusion import LateFusionModel #名稱之後會改

import torch
from  torch.utils.data import DataLoader , Dataset
from torch import nn , optim


#///////////////////////////////////////////////////////////////////////////#
                          # I n i t i a l i z e
#///////////////////////////////////////////////////////////////////////////#

audio_model = AudioCNN()
text_model = TextEncoder()
fusion_model = LateFusionModel( audio_dim=128 , text_dim=128 , output_dim=2 )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model = audio_model().to(device)
text_model = text_model().to(device)
fusion_model = fusion_model().to(device)


#///////////////////////////////////////////////////////////////////////////#
                          # L o a d  D a t a 
#///////////////////////////////////////////////////////////////////////////#
#這部分從 dataset load data，把三種data(audio.mp3 , lyric.txt , labels) 放進同一個 data_loader 中，之後把 data_loader 送進 fusion_model

train_dataset = MusicDataset( csv_path = 'TBD' , audio_dir = 'TBD' , lyric_path = 'TBD' , split = 'train' ) 
train_loader = DataLoader( train_dataset , batch_size = 'TBD' , shuffle = True )      #load training data (三合一)示意

val_dataset = MusicDataset( csv_path = 'TBD' , audio_dir = 'TBD' , lyric_path = 'TBD' , split = 'validate' ) 
val_loader = DataLoader( val_dataset , batch_size = 'TBD' , shuffle = True )          #load validate data (三合一)示意

test_dataset = MusicDataset( csv_path = 'TBD' , audio_dir = 'TBD' , lyric_path = 'TBD' , split = 'test' ) 
test_loader = DataLoader( test_dataset , batch_size = 'TBD' , shuffle = True )        #load test data (三合一)示意


#///////////////////////////////////////////////////////////////////////////#
                            # t r a i n i n g
#///////////////////////////////////////////////////////////////////////////#
# 這區域對 fusion_model 做 training，到時候把 for 迴圈留著然後在 fusion_model.py 做 train function

criterion = nn.MSELoss()
optimizer = optim.Adam( fusion_model.parameters() , lr=1e-4 )

print( "training session" )

train_losses = []
val_losses = []

EPOCHS = 10
for epoch in range(EPOCHS):
    audio_model.train()
    text_model.train()
    fusion_model.train()
    total_loss = 0
    
    for audios , lyrics , labels in train_loader:
        audios , lyrics , labels = audios.to(device), lyrics.to(device), labels.to(device)

        audios_feature = audio_model(audios) #得到 audio feature_vector，到時候作為 input 送給 fusion_model.train()
        lyrics_feature = text_model(lyrics)  #得到 lyric feature_vector，到時候作為 input 送給 fusion_model.train()

        predict_value = fusion_model( audios_feature , lyrics_feature )

        train_loss = criterion( predict_value , labels )
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_loss += train_loss.item()
        train_losses.append(train_loss)

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch+1} finished - Training Loss: {avg_loss:.4f}")


#///////////////////////////////////////////////////////////////////////////#
                            # v a l i d a t e 
#///////////////////////////////////////////////////////////////////////////#
# 這區域對 fusion_model 做 validating，到時候把 for 迴圈留著然後在 fusion_model.py 做 validate function

print( "validate session" )

audio_model.eval()
text_model.eval()
fusion_model.eval()

val_losses = []

total_val_loss = 0

with torch.no_grad():
    for audios, lyrics, labels in val_loader:
        audio = audio.to(device)
        lyrics = lyrics.to(device)
        labels = labels.to(device)

        audios_feature_val = audio_model(audio) #得到 audio feature_vector，到時候作為 input 送給 fusion_model.validate()
        lyrics_feature_val = text_model(lyrics) #得到 lyric feature_vector，到時候作為 input 送給 fusion_model.validate()

        predict_value_val = fusion_model(audios_feature_val, lyrics_feature_val)

        val_loss = criterion(predict_value_val, labels)
        val_losses.append(val_loss)
        total_val_loss += val_loss.item()

avg_val_loss = total_val_loss / len(val_loader)
print(f"Epoch: {epoch+1} finished - Val Loss: {avg_val_loss:.4f}")



#///////////////////////////////////////////////////////////////////////////#
                              # t e s t i n g 
#///////////////////////////////////////////////////////////////////////////#
# 這區域對 fusion_model 做 testing，到時候要搬進 fusion_model.py

print( "validate session" )

audio_model.eval()
text_model.eval()
fusion_model.eval()

test_losses = []

total_test_loss = 0

with torch.no_grad():
    for audios, lyrics, labels in test_loader:
        audio = audio.to(device)
        lyrics = lyrics.to(device)
        labels = labels.to(device)

        audios_feature_test = audio_model(audio) #得到 audio feature_vector，到時候作為 input 送給 fusion_model.validate()
        lyrics_feature_test = text_model(lyrics) #得到 lyric feature_vector，到時候作為 input 送給 fusion_model.validate()

        predict_value_test = fusion_model(audios_feature_test, lyrics_feature_test)

        test_loss = criterion(predict_value_test, labels)
        test_losses.append(test_loss)
        total_test_loss += test_loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print(f"Epoch: {epoch+1} finished - Test Loss: {avg_test_loss:.4f}")


#///////////////////////////////////////////////////////////////////////////#
                          # p l o t & o u t p u t
#///////////////////////////////////////////////////////////////////////////#
# output 一些圖表跟一些 csv 檔
