import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from tqdm import tqdm

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
        self.relu = nn.Relu()

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
    