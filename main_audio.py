from CNN import ( AduioCNN_solo , train , validate , test )

import torch
from  torch.utils.data import DataLoader , Dataset
from torch import nn , optim

import dataset
from dataset import ( Audio_solo_Dataset , tokenize , get_audios_paths ,
                      get_ids_and_labels , get_lyrics_paths , load_audio ,
                      load_lyric )


#///////////////////////////////////////////////////////////////////////////#
                          # I n i t i a l i z e
#///////////////////////////////////////////////////////////////////////////#

audio_model = AduioCNN_solo()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model = audio_model().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam( audio_model.parameters() , lr=1e-4 )

#///////////////////////////////////////////////////////////////////////////#
                       # L o a d  D a t a ( train )
#///////////////////////////////////////////////////////////////////////////#

train_ids , train_labels = get_ids_and_labels( csv_path='TBD' , Type='train' )
train_audios_paths = get_audios_paths( train_ids , audio_file_path='TBD' )

train_audios = load_audio( train_audios_paths )

train_dataset = Audio_solo_Dataset ( train_audios , train_labels )
train_loader = DataLoader( train_dataset , batch_size='TBD' , shuffle=True )


#///////////////////////////////////////////////////////////////////////////#
                     # L o a d  D a t a ( validate )
#///////////////////////////////////////////////////////////////////////////#

val_ids , val_labels = get_ids_and_labels( csv_path='TBD' , Type='validate' )
val_audios_paths = get_audios_paths( val_ids , audio_file_path='TBD' )

val_audios = load_audio( val_audios_paths )

val_dataset = Audio_solo_Dataset ( val_audios , val_labels )
val_loader = DataLoader( val_dataset , batch_size='TBD' , shuffle=True )


#///////////////////////////////////////////////////////////////////////////#
                       # L o a d  D a t a ( test )
#///////////////////////////////////////////////////////////////////////////#

test_ids , test_labels = get_ids_and_labels( csv_path='TBD' , Type='validate' )
test_audios_paths = get_audios_paths( test_ids , audio_file_path='TBD' )

test_audios = load_audio( test_audios_paths )

test_dataset = Audio_solo_Dataset ( test_audios , test_labels )
test_loader = DataLoader( test_dataset , batch_size='TBD' , shuffle=True )


#///////////////////////////////////////////////////////////////////////////#
                            # t r a i n i n g
#///////////////////////////////////////////////////////////////////////////#

train_losses = []
val_losses = []
best_val_loss = float('inf')

EPOCHS = 10
for epoch in range(EPOCHS): #epoch
    train_loss = train(audio_model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(audio_model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(audio_model.state_dict(), "best_model_audio.pt")
            print("Best model saved!")
   
    print ( "Training CNN .......... epoch =" , epoch , 
            " finished "   ", value_loss =" , val_loss ,
            " , value_accurate =" , val_acc )


#///////////////////////////////////////////////////////////////////////////#
                              # t e s t i n g 
#///////////////////////////////////////////////////////////////////////////#
test( audio_model , test_loader , criterion , device)