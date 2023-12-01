import torch
import random
from sklearn.model_selection import train_test_split

aa_map = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
    'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
    'V': 18, 'W': 19, 'Y': 20
}

def peptide_to_int(peptide):
    result = []
    for aa in peptide:
        result.append(aa_map[aa])
    return result


#tokenizer
#label tells the model if a peptide is an AMP or not
#(1 for AMP, 0 for non-AMP)
#i think we need to add [CLS] and [SEP] still.
def fa_to_list(fa_path, label):
    max_length = 300
    ids = []
    aa_sequences = []
    labels = []
    with open(fa_path, 'r') as file:
        for line in file:
            if line[0] == '>':
                ids.append(line[1:-1])
            else:
                peptide = line[:-1]
                new_rep = peptide_to_int(peptide) + [0]*(max_length-len(peptide))
                aa_sequences.append(new_rep)
                labels.append(label)
    aa_sequences = torch.tensor(aa_sequences, dtype=torch.long)
    return ids, aa_sequences, labels

amp_ids, amp_aa_sequences, amp_labels = fa_to_list('data/AMPs.fa', 1)
non_amp_ids, non_amp_aa_sequences, non_amp_labels = fa_to_list('data/Non-AMPs.fa', 0)
print(amp_ids[:3])
print(amp_aa_sequences[:3])
print(amp_labels[:3])
print(non_amp_ids[:3])
print(non_amp_aa_sequences[:3])
print(non_amp_labels[:3])

def generate_all_data(amp_aa_sequences, non_amp_aa_sequences, torch_save = False):
    all_data = []
    for aa in amp_aa_sequences:
        all_data.append((aa, 1))
    for aa in non_amp_aa_sequences:
        all_data.append((aa, 0))

    random.shuffle(all_data)
    sequences = []
    labels = []
    for tup in all_data:
        sequences.append(tup[0])
        labels.append(tup[1])

    if torch_save:
        torch.save(sequences, 'shuffled_sequences.pt')
        torch.save(labels, 'shuffled_labels.pt')
    return sequences, labels

def train_test_split(sequences, labels):
    # X_train, X_test, y_train, y_test = train_test_split(
    #     sequences, labels, test_size=.3
    # )
    test_size = 0.3
    split_idx = int(test_size*len(sequences))
    X_test = sequences[:split_idx]
    y_test = labels[:split_idx]
    X_train = sequences[split_idx:]
    y_train = labels[split_idx:]
    return X_train, X_test, y_train, y_test

shuffled_sequences, shuffled_labels = generate_all_data(amp_aa_sequences, non_amp_aa_sequences)
X_train, X_test, y_train, y_test = train_test_split(shuffled_sequences, shuffled_labels)
print(len(X_train))
print(len(X_test))
#sequences and labels contain all peptides in AMPs.fa and Non-AMPs.fa.
#additionally they are already shuffled.




#1085 AMP sequences and 58776 Non AMP sequences
print(len(amp_aa_sequences))
print(len(non_amp_aa_sequences))

#[abc, 0] []


# torch.save(amp_aa_sequences, 'amp_aa_sequences.pt')
# torch.save(non_amp_aa_sequences, 'non_amp_aa_sequences.pt')



