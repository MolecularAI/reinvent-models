from typing import List, Tuple

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.utils import data as tud

from reinvent_models.patformer.dto.patformer_batch_dto import PatformerBatchDTO
from reinvent_models.patformer.models.module.subsequent_mask import subsequent_mask


class PairedDataset(tud.Dataset):
    """Dataset that takes a list of (input, output) pairs."""
    # TODO check None for en_input, en_output
    def __init__(self, smiles_input: List[str], smiles_output: List[str], vocabulary, tokenizer):
        self.vocabulary = vocabulary
        self._tokenizer = tokenizer

        self._encoded_input_list = []
        for input_smi in smiles_input:
            tokenized_input = self._tokenizer.tokenize(input_smi)
            en_input = self.vocabulary.encode(tokenized_input)
            self._encoded_input_list.append(en_input)

        self._encoded_output_list = []
        for output_smi in smiles_output:
            tokenized_output = self._tokenizer.tokenize(output_smi)
            en_output = self.vocabulary.encode(tokenized_output)
            self._encoded_output_list.append(en_output)

    def __getitem__(self, i):
        en_input, en_output = self._encoded_input_list[i], self._encoded_output_list[i]
        return (torch.tensor(en_input, dtype=torch.long),
                torch.tensor(en_output, dtype=torch.long))  # pylint: disable=E1102

    def __len__(self):
        return len(self._encoded_input_list)

    @classmethod
    def collate_fn(cls, encoded_pairs) -> PatformerBatchDTO:
        """
        Turns a list of encoded pairs (input, target) of sequences and turns them into two batches.
        :param: A list of pairs of encoded sequences.
        :return: A tuple with two tensors, one for the input and one for the targets in the same order as given.
        """
        encoded_inputs, encoded_targets = list(zip(*encoded_pairs))
        collated_arr_source, src_mask = cls._mask_batch(encoded_inputs)
        collated_arr_target, trg_mask = cls._mask_batch(encoded_targets)

        # TODO: refactor the logic below
        trg_mask = trg_mask & Variable(subsequent_mask(collated_arr_target.size(-1)).type_as(trg_mask))
        trg_mask = trg_mask[:, :-1, :-1]  # save start token, skip end token

        dto = PatformerBatchDTO(collated_arr_source.cuda(),src_mask.cuda(),collated_arr_target.cuda(),trg_mask.cuda())
        return dto

    @staticmethod
    def _mask_batch(encoded_seqs: List) -> Tuple[Tensor, Tensor]:
        """
        Pads a batch.
        :param encoded_seqs: A list of encoded sequences.
        :return: A tensor with the sequences correctly padded and masked
        """
        # maximum length of input sequences
        max_length_source = max([seq.size(0) for seq in encoded_seqs])
        # padded source sequences with zeroes
        collated_arr_seq = torch.zeros(len(encoded_seqs), max_length_source, dtype=torch.long)
        for i, seq in enumerate(encoded_seqs):
            collated_arr_seq[i, : len(seq)] = seq
        # mask of source seqs
        seq_mask = (collated_arr_seq != 0).unsqueeze(-2)

        return collated_arr_seq, seq_mask
