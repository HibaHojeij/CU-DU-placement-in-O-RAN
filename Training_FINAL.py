import numpy as np
from torch.utils.data import Dataset
import torch
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from tqdm import tqdm  # For a nice progress bar!
import tensorboardX
import sys
import pandas as pd

# Set device
if torch.cuda.is_available():
    print('CUDA is available')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
input_size = 93  ##number of features
sequence_length = 100
hidden_size = 128
num_layers = 1
output_size = 13
num_classes = 13  ### is it the number of possible labels (we have 13 possibility of locations)
learning_rate = 0.005
batch_size = 60  # i choose it to be a divisor of number of samples
num_epochs = 1000

# Load the data
df = pd.read_csv("MASSIVE-DATASET_5000runs.csv")

# Extract the features
LOCATIONS = df[['LOCATION']]

features = df.drop(['LOCATION','user_loc_x','user_loc_y','loc_edge_x_1','loc_edge_x_2','loc_edge_x_3','loc_edge_y_1','loc_edge_y_2','loc_edge_y_3','loc_reg_x','loc_reg_y'],axis=1)
arr = df['RB'].values
arr_groups = arr.reshape(-1,100)
sum_groups = arr_groups.sum(axis=1)
repeated_sums = np.repeat(sum_groups,100)
features['Sum_RBS'] = repeated_sums / 400

arr = df['max_latency'].values
arr_groups = arr.reshape(-1,100)
sum_groups = arr_groups.sum(axis=1)
repeated_sums = np.repeat(sum_groups,100)

##Normalize the dataset
features['Sum_max_latency'] = repeated_sums / 10000
features = pd.get_dummies(features,columns=['RB','MCS','USer_association_BS','slice_type','priority'])
features['GOPS_required_DU'] = features['GOPS_required_DU'] / 30
features['GOPS_required_CU'] = features['GOPS_required_CU'] / 10
features['total_num_users'] = features['total_num_users'] / 100
features['max_latency'] = features['max_latency'] / 1000
features['TOTAL_GOPS_required_CU'] = features['TOTAL_GOPS_required_CU'] / 200
features['TOTAL_GOPS_required_DU'] = features['TOTAL_GOPS_required_DU'] / 500
features['GOPS_available_1'] = features['GOPS_available_1'] / 2000
features['GOPS_available_2'] = features['GOPS_available_2'] / 2000
features['GOPS_available_3'] = features['GOPS_available_3'] / 2000
features['GOPS_available_4'] = features['GOPS_available_4'] / 2000

features['rel_dist_Server_BS_0'] = features['rel_dist_Server_BS_0'] / 120
features['rel_dist_Server_BS_1'] = features['rel_dist_Server_BS_1'] / 120
features['rel_dist_Server_BS_2'] = features['rel_dist_Server_BS_2'] / 120
features['rel_dist_Server_BS_3'] = features['rel_dist_Server_BS_3'] / 120
features['rel_dist_Server_BS_4'] = features['rel_dist_Server_BS_4'] / 120
features['rel_dist_Server_BS_5'] = features['rel_dist_Server_BS_5'] / 120
features['rel_dist_Server_BS_6'] = features['rel_dist_Server_BS_6'] / 120
features['rel_dist_Server_BS_7'] = features['rel_dist_Server_BS_7'] / 120
features['rel_dist_Server_BS_8'] = features['rel_dist_Server_BS_8'] / 120
features['rel_dist_Server_BS_9'] = features['rel_dist_Server_BS_9'] / 120
features['rel_dist_Server_BS_10'] = features['rel_dist_Server_BS_10'] / 120
features['rel_dist_Server_BS_11'] = features['rel_dist_Server_BS_11'] / 120
features['rel_dist_Server_BS_12'] = features['rel_dist_Server_BS_12'] / 120
features['rel_dist_Server_BS_13'] = features['rel_dist_Server_BS_13'] / 120
features['rel_dist_Server_BS_14'] = features['rel_dist_Server_BS_14'] / 120
features['rel_dist_Server_BS_15'] = features['rel_dist_Server_BS_15'] / 120

features['rel_dist_ser_ser_0'] = features['rel_dist_ser_ser_0'] / 120
features['rel_dist_ser_ser_1'] = features['rel_dist_ser_ser_1'] / 120
features['rel_dist_ser_ser_2'] = features['rel_dist_ser_ser_2'] / 120
features['rel_dist_ser_ser_3'] = features['rel_dist_ser_ser_3'] / 120
features['rel_dist_ser_ser_4'] = features['rel_dist_ser_ser_4'] / 120
features['rel_dist_ser_ser_5'] = features['rel_dist_ser_ser_5'] / 120

features['link_latency_0_0'] = features['link_latency_0_0'] / 600
features['link_latency_0_1'] = features['link_latency_0_1'] / 600
features['link_latency_0_2'] = features['link_latency_0_2'] / 600
features['link_latency_0_3'] = features['link_latency_0_3'] / 600
features['link_latency_1_1'] = features['link_latency_1_1'] / 600
features['link_latency_1_2'] = features['link_latency_1_2'] / 600
features['link_latency_1_3'] = features['link_latency_1_3'] / 600
features['link_latency_2_2'] = features['link_latency_2_2'] / 600
features['link_latency_2_3'] = features['link_latency_2_3'] / 600
features['link_latency_3_3'] = features['link_latency_3_3'] / 600

features.to_csv('normalized_data.csv',index=False)


class CustomStarDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv("normalized_data.csv")
        self.df_labels = LOCATIONS
        self.df.drop(self.df.columns[0],axis=1,inplace=True)  ##drop the indexes of the dataframe
        # fill data from dataframe as a 3D tensor [40,32,19] to be the input of LSTM
        LIST_data = []
        LIST_target = []
        for BATCH in range(int(df.shape[0] / sequence_length) // int(batch_size)):  # to drop the last batch
            nested_list = []
            for index in range(batch_size):
                list = []
                for j in range(sequence_length):
                    list.append([self.df.iloc[j+sequence_length * index+batch_size * sequence_length * BATCH]])
                nested_list.append(list)  # collect each batch of data into nested_list
            nested_list = torch.tensor(nested_list).squeeze(dim=2).float()
            LIST_data.append(nested_list)

            # fill the targets as tensor
            nested_target = []
            for index in range(batch_size):
                list = []
                for j in range(sequence_length):
                    list.append([self.df_labels.iloc[j+sequence_length * index+batch_size * sequence_length * BATCH]])
                nested_target.append(list)
            nested_target = torch.tensor(nested_target).squeeze(dim=2).long()
            nested_target = nested_target.squeeze(dim=2)
            LIST_target.append(nested_target)

        self.train = LIST_data
        self.train_labels = LIST_target

        # print("traiiiiiiinn", len(self.train),"labelllllllllls",len(self.train_labels))  ## self.train containt the batched of data: list of 3D tensors)

    def __len__(self):
        return len(self.train)

    def __getitem__(self,idx):
        return self.train[idx],self.train_labels[idx]
        return self


DATA = CustomStarDataset()


def generate_mask(batch_sequence_lengths,max_sequence_length):
    mask = torch.zeros((len(batch_sequence_lengths),max_sequence_length),dtype=torch.float)
    for i,sequence_length in enumerate(batch_sequence_lengths):
        mask[i,:sequence_length] = 1
    return mask


batch_sequence_lengths = [20,40,60,80,100] * (batch_size // 5)  # decesending order because the dataset is sorted in descending before pack_padded

mask = generate_mask(batch_sequence_lengths,100)


class BRNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(BRNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2,num_classes)  # fully connected

    def forward(self,X,X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence

        h0 = torch.zeros(self.num_layers * 2,X.size(0),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2,X.size(0),self.hidden_size).to(device)

        batch_size,seq_len,_ = X.size()

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X,X_lengths,batch_first=True,enforce_sorted=False)

        # now run through LSTM
        X,_ = self.lstm(X,(h0,c0))

        # undo the packing operation
        X,_ = torch.nn.utils.rnn.pad_packed_sequence(X,batch_first=True)

        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1,X.shape[2])

        # run through actual linear layer
        X = self.fc(X)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_classes)
        X = X.view(batch_size,seq_len,num_classes)

        out = X
        return out


model = BRNN(input_size,hidden_size,num_layers,num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# Initialize the lists to store the accuracy and loss values
accuracies = []
losses = []

# Initialize a writer object for TensorBoard
writer = tensorboardX.SummaryWriter()

##START TRAINING
try:
    # Train Network
    for epoch in range(num_epochs):
        print("epoch:",epoch)
        num_correct = 0
        num_samples = 0
        for batch_idx,data in enumerate(tqdm(DATA.train)):
            data = data.to(device=device)
            targets = DATA.train_labels[batch_idx]
            targets = targets.to(device=device)

            # set model to training mode
            model.train()
            optimizer.zero_grad()
            scores = model(data,batch_sequence_lengths)  # shape: (batch_size, sequence_length, num_classes)
            # mask shape: (batch_size, sequence_length)

            # Convert the mask to have the same shape as scores
            if mask.ndim < 3:
                mask = mask.unsqueeze(-1).expand_as(scores).to(device=device)

            # Element-wise multiply scores with the mask
            scores = scores * mask

            # Calculate the loss using the masked scores
            # loss = criterion(scores, target)
            loss = criterion(scores.reshape(-1,num_classes),targets.reshape(-1))
            # print("TARGETT_reshaped", targets[batch_idx].reshape(-1).shape)
            # print("SCOREEE_reshaped", scores.reshape(-1,num_classes).shape)

            # backward
            loss.backward()

            # print the gradients of the model's parameters
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad.norm(2).item())

            # gradient descent update step/adam step
            optimizer.step()

            # Append the accuracy and loss values to the lists after every 10 batches
            if batch_idx % 10 == 0:
                # Set model to eval
                model.eval()
                with torch.no_grad():
                    num_correct = 0
                    num_samples = 0
                    for x,y in zip(DATA.train,DATA.train_labels):
                        x = x.to(device=device)
                        y = y.to(device=device)
                        # x = torch.flip(x, dims=(1,))
                        scores = model(x,batch_sequence_lengths)
                        _,max_index = torch.max(scores,dim=-1,keepdim=False)
                        predictions = max_index
                        for i in range(len(batch_sequence_lengths)):
                            NN = batch_sequence_lengths[i]
                            num_correct += (predictions[i,:NN] == y[i,:NN]).sum()
                            num_samples += NN
                    accuracy = (num_correct / num_samples) * 100
                    print("Accuracy:",accuracy)
                    # Element-wise multiply scores with the mask having already same shape of scores
                    scores = scores * mask
                    loss = criterion(scores.reshape(-1,num_classes),targets.reshape(-1))
                    print('loss',loss)
                    # Toggle model back to train
                    model.train()
    #### SAVE the trained model to file model.pt to use it when comparing with ILP model
    torch.save(model.state_dict(),'model.pt')
    print('END of Training')

except KeyboardInterrupt:
    print("Interrupted! Saving the model...")
    torch.save(model.state_dict(),"model-interrupted.pt")
    sys.exit(0)

def check_accuracy(DATA,model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x,y in zip(DATA.train,DATA.train_labels):
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x,batch_sequence_lengths)
            _,max_index = torch.max(scores,dim=-1,keepdim=False)  ## returns the max_value,max_index tuple element wise over the all sequences of the batch
            predictions = max_index
            num_correct += (predictions == y).sum()
    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(DATA,model) * 100:.2f}")
