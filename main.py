from CNN import AudioCNN
#from BERT import TextEncoder
#from Fusion import LateFusionModel

import torch
from  torch.utils.data import DataLoader , Dataset
from torch import nn , optim


#///////////////////////////////////////////////////////////////////////////#
                          # I n i t i a l i z e
#///////////////////////////////////////////////////////////////////////////#

audio_model = AudioCNN()
text_model = TextEncoder()
fusion_model = LateFusionModel( audio_dim=128 , text_dim=768 , output_dim=2 )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model = audio_model().to(device)
text_model = text_model().to(device)
fusion_model = fusion_model().to(device)

train_dataset = MusicDataset( csv_path = 'TBD' , audio_dir = 'TBD' , lyric_path = 'TBD' , split = 'train' )
train_loader = DataLoader( train_dataset , batch_size = 'TBD' , shuffle = True )

val_dataset = MusicDataset( csv_path = 'TBD' , audio_dir = 'TBD' , lyric_path = 'TBD' , split = 'validate' )
val_loader = DataLoader( val_dataset , batch_size = 'TBD' , shuffle = True )

test_dataset = MusicDataset( csv_path = 'TBD' , audio_dir = 'TBD' , lyric_path = 'TBD' , split = 'test' )
test_loader = DataLoader( test_dataset , batch_size = 'TBD' , shuffle = True )

criterion = nn.MSELoss()
optimizer = optim.Adam( fusion_model.parameters() , lr=1e-4 )



#///////////////////////////////////////////////////////////////////////////#
                            # t r a i n i n g
#///////////////////////////////////////////////////////////////////////////#

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

        audios_feature = audio_model(audios)
        lyrics_feature = text_model(lyrics)

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

        audios_feature_val = audio_model(audio)
        lyrics_feature_val = text_model(lyrics)

        predict_value_val = fusion_model(audios_feature_val, lyrics_feature_val)

        val_loss = criterion(predict_value_val, labels)
        val_losses.append(val_loss)
        total_val_loss += val_loss.item()

avg_val_loss = total_val_loss / len(val_loader)
print(f"Epoch: {epoch+1} finished - Val Loss: {avg_val_loss:.4f}")



#///////////////////////////////////////////////////////////////////////////#
                              # t e s t i n g 
#///////////////////////////////////////////////////////////////////////////#