import torch
import torch.nn as nn

MAX_LEN = 2000

class LateFusionModel(nn.Module):
    def __init__(self, vocab_size , audio_dim = 128 , text_dim=128, max_len=MAX_LEN , output_dim=2):
        super(LateFusionModel, self).__init__()

        self.audio_dim = audio_dim
        self.lyric_dim = text_dim
        self.embed_dim = 50
        conv_output_len = max_len // 2

        self.relu = nn.ReLU()

        # Audio 分支
        self.audio_branch_conv = nn.Sequential(
            nn.Conv2d( in_channels=1 , out_channels=16 , kernel_size=3 , padding=1 ),
            nn.ReLU(),
            nn.MaxPool2d( kernel_size=2 ), # 16 , 16 , 64 ,512
            nn.Conv2d( in_channels=16 , out_channels=32 , kernel_size=3 , padding=1 ),
            nn.ReLU(),
            nn.MaxPool2d( kernel_size=2 ), # 16 , 32 , 32 ,256
            nn.Conv2d( in_channels=32 , out_channels=64 , kernel_size=3 , padding=1 ),
            nn.ReLU(),
            nn.MaxPool2d( kernel_size=2 ), # 16 , 64 , 16 , 128
        )

        self.audio_branch_fc = nn.Linear(64 * 16 * 128 ,self.audio_dim)
        

        # Text 分支
        self.embedding = nn.Embedding(vocab_size, self.embed_dim, padding_idx=0)
        
        self.text_branch_conv = nn.Sequential(
            nn.Conv1d(self.embed_dim, 100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.text_branch_fc = nn.Sequential(
            nn.Linear(100 * conv_output_len, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, text_dim),
        )

        # 最終融合層（將 audio + text 預測結合再微調）
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.audio_dim + self.lyric_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.Linear(128, output_dim),  
            nn.Sigmoid()
        )

    def forward(self, audios, texts):
        audio_out = self.audio_branch_conv(audios)
        audio_out = audio_out.view( audio_out.size(0) , -1 )   
        audio_feature = self.audio_branch_fc(audio_out)

        text_out = self.embedding(texts)         
        text_out = text_out.permute(0, 2, 1)          
        text_out = self.text_branch_conv(text_out)                
        text_out = text_out.view(text_out.size(0), -1) 
        text_feature = self.text_branch_fc(text_out) 

        fused_input = torch.cat([audio_feature, text_feature], dim=1)  
        output = self.fusion_layer(fused_input)                

        return output
