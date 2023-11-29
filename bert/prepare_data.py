import torch

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
def fa_to_list(fa_path, label, include_label=False):
    max_length = 300
    ids = []
    aa_sequences = []
    with open(fa_path, 'r') as file:
        for line in file:
            if line[0] == '>':
                ids.append(line[1:-1])
            else:
                peptide = line[:-1]
                new_rep = peptide_to_int(peptide) + [0]*(max_length-len(peptide))
                if include_label:
                    new_rep.append(label)
                aa_sequences.append(new_rep)
    aa_sequences = torch.tensor(aa_sequences, dtype=torch.long)
    return ids, aa_sequences

amp_ids, amp_aa_sequences = fa_to_list('data/AMPs.fa', 1)
non_amp_ids, non_amp_aa_sequences = fa_to_list('data/Non-AMPs.fa', 0)
print(amp_ids[:3])
print(amp_aa_sequences[:3])
print(non_amp_ids[:3])
print(non_amp_aa_sequences[:3])

torch.save(amp_aa_sequences, 'amp_aa_sequences.pt')
torch.save(non_amp_aa_sequences, 'non_amp_aa_sequences.pt')



