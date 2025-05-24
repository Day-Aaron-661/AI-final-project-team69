import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd


#///////////////////////////////////////////////////////////////////////////#
                          # A u d i o C N N
#///////////////////////////////////////////////////////////////////////////#

class AudioCNN(nn.Module):
    def __init__(self , n_mels = 256 , fixed_frame = 1024 ):
        super(AudioCNN, self).__init__()
        self.n_mels = n_mels
        self.fixed_frame = fixed_frame

        self.conv1 = nn.Conv2d( in_channels=1 , out_channels=16 , kernel_size=3 , padding=1 )
        self.pool1 = nn.MaxPool2d( kernel_size=2 )

        self.conv2 = nn.Conv2d( in_channels=16 , out_channels=32 , kernel_size=3 , padding=1 )
        self.pool2 = nn.MaxPool2d( kernel_size=2 )

        self.conv3 = nn.Conv2d( in_channels=32 , out_channels=64 , kernel_size=3 , padding=1 )
        self.pool3 = nn.MaxPool2d( kernel_size=2)

        self.fc1 = nn.Linear(64 * 32 * 128 ,2048)
        self.fc2 = nn.Linear(2048,128)

        self.relu = nn.ReLU()

    def forward ( self , x ):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = x.view( x.size(0) , -1 )    
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    


#///////////////////////////////////////////////////////////////////////////#
                          # M p 3 to mel-spectrogram 
#///////////////////////////////////////////////////////////////////////////#

def mp3_to_mel( file_path , sr=22050 , n_mels=256, fixed_frames=1024 , duration=30 ):
    y, _ = librosa.load(file_path , sr = sr , duration = duration )
    mel = librosa.feature.melspectrogram( y = y , sr = sr , n_mels = n_mels )
    mel = librosa.power_to_db(mel , ref=np.max )
    mel = ( mel - mel.mean()) / (mel.std() + 1e-6 )

    if mel.shape[1] < fixed_frames:
        pad_nums = fixed_frames - mel.shape[1]
        mel = np.pad( mel , ((0,0) , (0,pad_nums)) , mode="constant" , constant_values=(0,0))
    else:
        mel = mel[ : , :fixed_frames]

    mel_tensor = torch.tensor(mel).unsqueeze(0).float()

    return mel_tensor



#///////////////////////////////////////////////////////////////////////////#
                          # A u d i o C N N solo 
#///////////////////////////////////////////////////////////////////////////#


class AudioCNN_solo(nn.Module):
    def __init__(self , n_mels = 256 , fixed_frame = 1024 ):
        super(AudioCNN_solo, self).__init__()
        self.n_mels = n_mels
        self.fixed_frame = fixed_frame

        self.conv1 = nn.Conv2d( in_channels=1 , out_channels=16 , kernel_size=3 , padding=1 )
        self.pool1 = nn.MaxPool2d( kernel_size=2 ) # 16 , 16 , 128 ,512

        self.conv2 = nn.Conv2d( in_channels=16 , out_channels=32 , kernel_size=3 , padding=1 )
        self.pool2 = nn.MaxPool2d( kernel_size=2 ) # 16 , 32 , 64 ,256

        self.conv3 = nn.Conv2d( in_channels=32 , out_channels=64 , kernel_size=3 , padding=1 )
        self.pool3 = nn.MaxPool2d( kernel_size=2 ) # 16 , 64 , 32 , 128

        self.fc1 = nn.Linear(64 * 32 * 128 ,2048)
        self.fc2 = nn.Linear(2048,128)
        self.fc3 = nn.Linear(128,2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward ( self , x ):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = x.view( x.size(0) , -1 )    
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        x = self.sigmoid(x)
        return x
    


def train(model: AudioCNN_solo, train_loader: DataLoader, criterion, optimizer, device) -> float:
    model.train()

    total_loss = 0.0
    progress = tqdm( train_loader , desc= 'Training', leave= False)

    for audios , labels in progress:
        labels = labels.to(device)
        audios = audios.to(device)

        optimizer.zero_grad()
        outputs = model(audios)

        loss = criterion(outputs , labels)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * audios.size(0)
    
    avg_loss = total_loss / len( train_loader.dataset)

    return avg_loss



def validate(model: AudioCNN_solo, val_loader: DataLoader, criterion, device) -> float:
    model.eval()
    
    total_loss = 0.0
    
    with torch.no_grad():
        progress = tqdm( val_loader , desc= 'validating', leave= False)

        for audios , labels in progress:
            labels = labels.to(device)
            audios = audios.to(device)

            outputs = model(audios)

            loss = criterion(outputs , labels)
            total_loss += loss.item() * audios.size(0)
    
    avg_loss = total_loss / len( val_loader.dataset)
    
    return avg_loss



def test(model: AudioCNN_solo, test_loader: DataLoader, criterion, device):
    model.eval()
    results = []

    progress = tqdm( test_loader , desc='Testing' , leave=False )

    for audios , audio_ids in progress:
        audios = audios.to(device)

        outputs = model(audios)

        energies = outputs[:, 0].tolist() 
        valences = outputs[:, 1].tolist()
        audio_ids = audio_ids.tolist()

        for audio_id, energy, valence in zip(audio_ids, energies, valences):
            results.append([audio_id, energy, valence])

    test_data = pd.DataFrame( results , columns=['id' , 'energy' , 'valence'])
    test_data.to_csv('audio_predictions.csv' , index=False)

    print(f"Predictions saved to 'audio_predictions.csv'")
    return
