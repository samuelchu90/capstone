#Used GPT-4
from transformers import BertTokenizer, BertModel, BertConfig
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#CLS, training, test split, unbalanced dataset AMPS vs nonAMPS

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(768, 2),
    )

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased")

model = BertModel(config)
model.add_module('mlp', MLP())

# Load the pre-trained weights
# Edit this line accordingly if using a GPU
model_arc = torch.load('data/bert.bin', map_location=torch.device('cpu'))

new_state = {}
for k, v in model_arc['state_dict'].items():
    if k.startswith('bert'):
        k = k.split('.', 1)[1]
    new_state[k] = v
model_arc['state_dict'] = new_state

model_arc['state_dict']['mlp.layers.0.weight'] = model_arc['state_dict']['mlp.weight']
model_arc['state_dict']['mlp.layers.0.bias'] = model_arc['state_dict']['mlp.bias']
del model_arc['state_dict']['mlp.weight']
del model_arc['state_dict']['mlp.bias']

print(config)

model.load_state_dict(model_arc['state_dict'], strict=True)
print(model)

def threshold(tensor):
    return (tensor>0.5).float()

class FullModel(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(768, 1)

    #add threshold if we only want forward pass, but for training we don't want threshold.
    #forward always wants to return a float.
    def forward(self, x):
        mask = (x>0).float()
        pooler_output = self.bert(x, attention_mask=mask).pooler_output
        linear_output = self.linear(pooler_output)
        return torch.sigmoid(linear_output)

    def predict(self, x):
        pooler_output = self.bert(x).pooler_output
        linear_output = self.linear(pooler_output)
        sigmoid_output = torch.sigmoid(linear_output)
        return threshold(sigmoid_output)

amp_aa_sequences = torch.load('data/amp_aa_sequences.pt')
non_amp_aa_sequences = torch.load('data/non_amp_aa_sequences.pt')
smaller_amp_aa_sequences = amp_aa_sequences[:3]

full_model = FullModel(model)
outputs = full_model(smaller_amp_aa_sequences)

print(outputs)
#print(outputs.pooler_output.shape, outputs.last_hidden_state.shape)

 #batch size x #tokens x #d_model multiply this matrix by #d_model x 1
#binary cross entropy


#DataLoader
class AMPDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label

#Backprop
sequences = torch.load('data/shuffled_sequences.pt')
labels = torch.load('data/shuffled_labels.pt')

print('here')
print(len(sequences))

dataset = AMPDataset(sequences, labels)
dataloader = DataLoader(dataset, batch_size=64)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(full_model.parameters(), lr=.00002)

num_epochs = 1
for epoch in range(num_epochs):
    for sequences, labels in dataloader:
        sequences = sequences.int()
        labels = labels.float()
        optimizer.zero_grad()
        outputs = full_model(sequences)
        outputs = outputs.squeeze(1)
        print(outputs.shape)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
