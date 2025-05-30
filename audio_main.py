from audio_model import ( AudioCNN , train , validate , test )
from make_mel import process_all_mp3_to_pt
import torch
from  torch.utils.data import DataLoader , Dataset
from torch import nn , optim
from tqdm import tqdm

from dataset import ( Audio_Dataset , get_audios_paths ,
                      get_ids_and_labels ,  load_audio ,)


#///////////////////////////////////////////////////////////////////////////#
                          # I n i t i a l i z e
#///////////////////////////////////////////////////////////////////////////#

audio_model = AudioCNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model = audio_model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam( audio_model.parameters() , lr=1e-3 )

# process_all_mp3_to_pt( audio_dir='data//audio' )

#///////////////////////////////////////////////////////////////////////////#
                       # L o a d  D a t a ( train )
#///////////////////////////////////////////////////////////////////////////#

print("Loading training data...")

train_ids , train_labels = get_ids_and_labels( csv_path='data//labels.csv' , Type='train' )
train_audios_paths = get_audios_paths( train_ids , audio_file_path='data//audio' )

train_audios = load_audio( train_audios_paths )

train_dataset = Audio_Dataset ( train_audios , train_labels )
train_loader = DataLoader( train_dataset , batch_size=16 , shuffle=True )


#///////////////////////////////////////////////////////////////////////////#
                     # L o a d  D a t a ( validate )
#///////////////////////////////////////////////////////////////////////////#
print("Loading validation data...")

val_ids , val_labels = get_ids_and_labels( csv_path='data//labels.csv' , Type='validate' )
val_audios_paths = get_audios_paths( val_ids , audio_file_path='data//audio' )

val_audios = load_audio( val_audios_paths )

val_dataset = Audio_Dataset ( val_audios , val_labels )
val_loader = DataLoader( val_dataset , batch_size=16 , shuffle=False )


#///////////////////////////////////////////////////////////////////////////#
                       # L o a d  D a t a ( test )
#///////////////////////////////////////////////////////////////////////////#

print("Loading test data...")
test_ids , test_labels = get_ids_and_labels( csv_path='data//labels.csv' , Type='test' )
test_audios_paths = get_audios_paths( test_ids , audio_file_path='data//audio' )

test_audios = load_audio( test_audios_paths )

test_dataset = Audio_Dataset ( test_audios , test_labels )
test_loader = DataLoader( test_dataset , batch_size=16 , shuffle=False )


#///////////////////////////////////////////////////////////////////////////#
                            # t r a i n i n g
#///////////////////////////////////////////////////////////////////////////#

print("Starting training...")
train_losses = []
val_losses = []
best_val_loss = float('inf')

EPOCHS = 30
for epoch in range(EPOCHS):
    train_loss = train(audio_model, train_loader, criterion, optimizer, device)
    val_loss = validate(audio_model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(audio_model.state_dict(), "best_model_audio.pt")
        print("Best model saved!")
   
    print("Training CNN .......... epoch =", epoch, 
          " finished ",", train_loss =", train_loss,  ", value_loss =", val_loss,)


#///////////////////////////////////////////////////////////////////////////#
                              # t e s t i n g 
#///////////////////////////////////////////////////////////////////////////#
test( audio_model , test_loader , criterion , device)
