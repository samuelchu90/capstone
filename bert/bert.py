#Used GPT-4
from transformers import BertTokenizer, BertModel, BertConfig
import torch
from torch import nn

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

class FullModel(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(768, 1)

    def forward(self, x):
        mask = (x>0).float()
        pooler_output = self.bert(x, attention_mask=mask).pooler_output
        return self.linear(pooler_output)

amp_aa_sequences = torch.load('data/amp_aa_sequences.pt')
non_amp_aa_sequences = torch.load('data/non_amp_aa_sequences.pt')
smaller_amp_aa_sequences = amp_aa_sequences[:1]

full_model = FullModel(model)
outputs = full_model(smaller_amp_aa_sequences)

print(outputs)
#print(outputs.pooler_output.shape, outputs.last_hidden_state.shape)

 #batch size x #tokens x #d_model multiply this matrix by #d_model x 1
#binary cross entropy