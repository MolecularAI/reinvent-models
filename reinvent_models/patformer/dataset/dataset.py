# coding=utf-8

"""
Implementation of a SMILES dataset.
"""
import torch
import torch.utils.data as tud


class Dataset(tud.Dataset):
    """Custom PyTorch Dataset
    """

    def __init__(self, smiles_list, vocabulary, tokenizer):
        """
        :param smiles_list: A list with SMILES strings
        :param vocabulary: used to encode tokens
        :param tokenizer: used to tokenize smiles
        """
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer

        self._encoded_list = []
        for smi in smiles_list:
            tokenized = self._tokenizer.tokenize(smi)
            enc = self._vocabulary.encode(tokenized)

            if enc is not None:
                self._encoded_list.append(enc)

    def __getitem__(self, i):
        return torch.tensor(self._encoded_list[i], dtype=torch.long)

    def __len__(self):
        return len(self._encoded_list)

    @classmethod
    def collate_fn(cls, encoded_seqs):
        # maximum length of input sequences
        max_length_source = max([seq.size(0) for seq in encoded_seqs])
        # padded source sequences with zeroes
        collated_arr_source = torch.zeros(len(encoded_seqs), max_length_source, dtype=torch.long)
        for i, seq in enumerate(encoded_seqs):
            collated_arr_source[i, : len(seq)] = seq
        # mask of source seqs
        src_mask = (collated_arr_source !=0).unsqueeze(-2)

        return collated_arr_source.cuda(), src_mask.cuda()