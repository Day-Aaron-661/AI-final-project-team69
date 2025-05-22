import torch
from torch import nn

class FusionModel(nn.Module):
    def __init__(self, audio_dim=128, text_dim=128, output_dim=2):
        super(FusionModel, self).__init__()
        
        self.fc1 = nn.Linear(audio_dim + text_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)  

    def forward(self, audio_feat, text_feat):
        
        fused = torch.cat((audio_feat, text_feat), dim=1) 
        
        x = self.fc1(fused)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        output = self.fc3(x)  
        
        return output
