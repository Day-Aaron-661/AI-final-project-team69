import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd


#///////////////////////////////////////////////////////////////////////////#
                          # A u d i o C N N
#///////////////////////////////////////////////////////////////////////////#

class AudioCNN(nn.Module):
    def __init__(self , n_mels = 128 , fixed_frame = 1024 ):
        super(AudioCNN, self).__init__()
        self.n_mels = n_mels
        self.fixed_frame = fixed_frame

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(64 * 16 * 128, 128)
        self.dropout_fc = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward ( self , x ):
        x = self.conv1(x)
        x = self.bn1(x)##
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)##

        x = self.conv2(x)
        x = self.bn2(x)##
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)##

        x = self.conv3(x)
        x = self.bn3(x)##
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)##
        # print("Shape after pool3:", x.shape) 


        x = x.view( x.size(0) , -1 )    

        feature = self.fc1(x) # 128
        output = self.relu(feature) 
        output = self.dropout_fc(output)##
        output = self.fc2(output)

        # output = self.sigmoid(output) # 2
        return feature , output
    

def train(model: AudioCNN, train_loader: DataLoader, criterion, optimizer, device) -> float:
    model.train()

    total_loss = 0.0
    progress = tqdm( train_loader , desc= 'Training', leave= False)

    for audios , labels in progress:
        labels = labels.to(device)
        audios = audios.to(device)

        optimizer.zero_grad()
        _ , outputs = model(audios)

        loss = criterion(outputs , labels)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * audios.size(0)
    
    avg_loss = total_loss / len( train_loader.dataset)

    return avg_loss



def validate(model: AudioCNN, val_loader: DataLoader, criterion, device) -> float:
    model.eval()
    
    total_loss = 0.0
    
    with torch.no_grad():
        progress = tqdm( val_loader , desc= 'validating', leave= False)

        for audios , labels in progress:
            labels = labels.to(device)
            audios = audios.to(device)

            _ , outputs = model(audios)

            loss = criterion(outputs , labels)
            total_loss += loss.item() * audios.size(0)
    
    avg_loss = total_loss / len( val_loader.dataset)
    
    return avg_loss



def test(model: AudioCNN, test_loader: DataLoader, criterion, device):
    model.eval()
    results = []

    progress = tqdm( test_loader , desc='Testing' , leave=False )

    for audios , audio_ids in progress:
        audios = audios.to(device)

        _ , outputs = model(audios)

        energies = outputs[:, 0].tolist() 
        valences = outputs[:, 1].tolist()
        audio_ids = audio_ids.tolist()

        for audio_id, energy, valence in zip(audio_ids, energies, valences):
            results.append([audio_id, energy, valence])

    test_data = pd.DataFrame( results , columns=['id' , 'energy' , 'valence'])
    test_data.to_csv('audio_predictions.csv' , index=False)

    print(f"Predictions saved to 'audio_predictions.csv'")
    return
