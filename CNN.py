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

class AduioCNN():
    def __init__(self , n_mels = 128 , fixed_frame = 128 ):
        super(AduioCNN, self).__init__()
        self.n_mels = n_mels
        self.fixed_frame = fixed_frame

        self.conv1 = nn.conv2d( in_channels=1 , out_channels=16 , kernel_size=3 , padding=1 )
        self.pool1 = nn.MaxPool2d( kernel_size=2 )

        self.conv2 = nn.conv2d( in_channels=16 , out_channels=32 , kernel_size=3 , padding=1 )
        self.pool2 = nn.MaxPool2d( kernel_size=2 )

        self.conv3 = nn.conv2d( in_chnanels=32 , out_channels=64 , kernel_size=3 , padding=1 )
        self.pool3 = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(64,128)
        self.relu = F.Relu()

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
        x = self.fc(x)

        return x
    

#///////////////////////////////////////////////////////////////////////////#
                          # M p 3 to mel-spectrogram 
#///////////////////////////////////////////////////////////////////////////#

def mp3_to_mel( file_path , sr=22050 , n_mels=128, fixed_frames=128 , duration=30 ):
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


class AduioCNN_solo():
    def __init__(self , n_mels = 128 , fixed_frame = 128 ):
        super(AduioCNN, self).__init__()
        self.n_mels = n_mels
        self.fixed_frame = fixed_frame

        self.conv1 = nn.conv2d( in_channels=1 , out_channels=16 , kernel_size=3 , padding=1 )
        self.pool1 = nn.MaxPool2d( kernel_size=2 )

        self.conv2 = nn.conv2d( in_channels=16 , out_channels=32 , kernel_size=3 , padding=1 )
        self.pool2 = nn.MaxPool2d( kernel_size=2 )

        self.conv3 = nn.conv2d( in_chnanels=32 , out_channels=64 , kernel_size=3 , padding=1 )
        self.pool3 = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128,2)
        self.relu = F.Relu()

    def forward ( self , x ):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = x.view( x.size(0) , -1 )    
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
def train(model: AduioCNN_solo, train_loader: DataLoader, criterion, optimizer, device)->float:
    model.train()

    total_loss = 0.0
    progress = tqdm( train_loader , desc= 'Training', leave= False)

    for audio , labels in progress:
        labels = labels.to(device)
        audio = audio.to(device)

        optimizer.zero_grad()
        outputs = model(audio)

        loss = criterion(outputs , labels)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * audio.size(0)
    
    avg_loss = total_loss / len( train_loader.dataset)

    return avg_loss


def validate(model: AduioCNN_solo, val_loader: DataLoader, criterion, device)->float:
    model.eval()
    
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        progress = tqdm( val_loader , desc= 'validating', leave= False)

        for images , labels in progress:
            labels = labels.to(device)
            images = images.to(device)

            outputs = model(images)

            loss = criterion(outputs , labels)
            total_loss += loss.item() * images.size(0)

            trash , predict = torch.max(outputs , 1)
            
            comparison = ( predict == labels )
            tensor_sum = comparison.sum()
            correct += tensor_sum.item()
    
    avg_loss = total_loss / len( val_loader.dataset)
    
    return avg_loss

def test(model: AduioCNN_solo, test_loader: DataLoader, criterion, device):
    
    model.eval()
    results = []

    progress = tqdm( test_loader , desc='Testing' , leave=False )

    for audio , audio_id in progress:
        audio = audio.to(device)

        outputs = model(audio)

        trash , predict = torch.max ( outputs , 1 )
        
        results.extend(zip( audio_id , predict.cpu().numpy() ) )

    test_data = pd.DataFrame( results , columns=['id' , 'prediction'])
    test_data.to_csv('CNN_solo.csv' , index=False)

    print(f"Predictions saved to 'CNN.csv'")
    return
