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

model.load_state_dict(model_arc['state_dict'], strict=True)
print(model)
#model.load_state_dict(torch.load('bert.bin', map_location=torch.device('cpu'))) #errorr


amp_aa_sequences = torch.load('data/amp_aa_sequences.pt')
non_amp_aa_sequences = torch.load('data/non_amp_aa_sequences.pt')
smaller_amp_aa_sequences = amp_aa_sequences[:5]
with torch.no_grad():
    outputs = model(smaller_amp_aa_sequences)

first_output = outputs[0]
print(first_output)