import librosa
import numpy as np
import torch
import os

#///////////////////////////////////////////////////////////////////////////#
                          # M p 3 to mel-spectrogram 
#///////////////////////////////////////////////////////////////////////////#

def mp3_to_mel( file_path , sr=22050 , n_mels=128, fixed_frames=1024 , duration=30 ):
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

def process_all_mp3_to_pt(audio_dir, save_dir='mel_tensors', sr=22050, n_mels=128, fixed_frames=1024, duration=30):
    os.makedirs(save_dir, exist_ok=True)

    for filename in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, filename)
        pt_filename = filename.replace(".mp3", ".pt")
        if filename.endswith('.mp3') and (not os.path.exists(pt_filename)):
            mel_tensor = mp3_to_mel(file_path, sr, n_mels, fixed_frames, duration)

            save_path = os.path.join(save_dir, pt_filename)

            torch.save(mel_tensor, save_path)
