import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import r2_score

#///////////////////////////////////////////////////////////////////////////#
                          # A u d i o C N N
#///////////////////////////////////////////////////////////////////////////#

# class AudioCNN(nn.Module):
#     def __init__(self, n_mels=128, fixed_frames=1024, output_dim=2):
#         super(AudioCNN, self).__init__()
#         self.n_mels = n_mels
#         self.fixed_frames = fixed_frames

#         # 1D Conv over time (input shape: batch, channels=n_mels, time=fixed_frames)
#         self.conv1 = nn.Conv1d(in_channels=n_mels, out_channels=32, kernel_size=8, stride=1)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=8, stride=1)
#         self.bn2 = nn.BatchNorm1d(16)
#         self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

#         # Flatten and Fully Connected Layers
#         # You'll need to infer the correct input size based on your input fixed_frames size.
#         dummy_input = torch.zeros(1, n_mels, fixed_frames)
#         out = self.forward_feature(dummy_input)
#         feature_dim = out.shape[1]

#         self.fc1 = nn.Linear(feature_dim, 64)
#         self.fc2 = nn.Linear(64, output_dim)

#     def forward_feature(self, x):
#         # x shape: (batch, mel, time)
#         x = x.squeeze(1)  # from [B, 1, 128, 1024] → [B, 128, 1024]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.pool1(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.pool2(x)

#         x = x.view(x.size(0), -1)  # Flatten
#         return x

#     def forward(self, x):
#         x = self.forward_feature(x)
#         feature = self.fc1(x)
#         output = self.fc2(F.relu(feature))
#         return feature, output


class AudioCNN(nn.Module):
    def __init__(self , n_mels = 128 , fixed_frame = 1024 ):
        super(AudioCNN, self).__init__()
        self.n_mels = n_mels
        self.fixed_frame = fixed_frame

        self.conv1 = nn.Conv2d( in_channels=1 , out_channels=16 , kernel_size=3 , padding=1 )
        self.pool1 = nn.MaxPool2d( kernel_size=2 ) # 16 , 16 , 64 ,512

        self.conv2 = nn.Conv2d( in_channels=16 , out_channels=32 , kernel_size=3 , padding=1 )
        self.pool2 = nn.MaxPool2d( kernel_size=2 ) # 16 , 32 , 32 ,256

        self.conv3 = nn.Conv2d( in_channels=32 , out_channels=64 , kernel_size=3 , padding=1 )
        # self.pool3 = nn.MaxPool2d( kernel_size=2 ) # 16 , 64 , 16 , 128
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))  # 輸出會變成 [B, 64, 4, 4]

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        # self.fc1 = nn.Linear(64 * 16 * 128 ,128)

        self.fc2 = nn.Linear(128,2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.2)

    def forward ( self , x ):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        # x = self.drop(x)

        x = x.view( x.size(0) , -1 )    

        feature = self.fc1(x) # 128
        output = self.drop(feature)
        output = self.relu(output) 

        output = self.fc2(output)

        output = self.sigmoid(output) # 2
        return feature , output

# class AudioCNN(nn.Module):
#     def __init__(self, n_mels=128, fixed_frames=1024, hidden_dim=128, num_layers=1, output_dim=2):
#         super(AudioCNN, self).__init__()

#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),

#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#         )

#         self.rnn_input_dim = 32 * (n_mels // 4)  # for example: 32 * 32 = 1024
#         self.rnn = nn.GRU(
#             input_size=self.rnn_input_dim, 
#             hidden_size=hidden_dim, 
#             num_layers=num_layers, 
#             batch_first=True, 
#             bidirectional=True
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, output_dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.cnn(x)                       # [B, 32, H, W]
#         b, c, h, w = x.shape
#         x = x.permute(0, 3, 1, 2)             # [B, W, C, H]
#         x = x.contiguous().view(b, w, c * h)  # [B, T, rnn_input_dim]

#         rnn_out, _ = self.rnn(x)              # [B, T, hidden_dim * 2]
#         feature = rnn_out[:, -1, :]           # [B, hidden_dim * 2]

#         output = self.fc(feature)             # [B, output_dim]

#         return feature, output


def train(model: AudioCNN, train_loader: DataLoader, criterion, optimizer, device) -> float:
    model.train()

    total_loss = 0.0
    progress = tqdm( train_loader , desc= 'Training', leave= False)

    for audios , labels , ids in progress:
        labels = labels.to(device)
        audios = audios.to(device)

        optimizer.zero_grad()
        _,outputs = model(audios)

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

        for audios , labels , ids in progress:
            labels = labels.to(device)
            audios = audios.to(device)

            _,outputs = model(audios)

            loss = criterion(outputs , labels)
            total_loss += loss.item() * audios.size(0)
    
    avg_loss = total_loss / len( val_loader.dataset)
    
    return avg_loss



def test(model: AudioCNN, test_loader: DataLoader, criterion, device):
    model.eval()
    results = []
    y_true_energy = []
    y_pred_energy = []

    y_true_valence = []
    y_pred_valence = []
    progress = tqdm( test_loader , desc='Testing' , leave=False )

    for audios , labels , audio_ids in progress:
        audios = audios.to(device)
        labels = labels.to(device)

        _,outputs = model(audios)

        pred_energy = outputs[:, 0].detach().cpu().numpy()
        pred_valence = outputs[:, 1].detach().cpu().numpy()

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
    test_data.to_csv('audio_predictions.csv' , index=False)

    print(f"Predictions saved to 'audio_predictions.csv'")
    return
