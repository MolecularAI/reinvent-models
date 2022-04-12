from typing import List
import torch
import torch.utils.data as tud

from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.patformer.dataset.paired_dataset import PairedDataset
from reinvent_models.patformer.dto import BatchLikelihoodDTO
from reinvent_models.patformer.dto.sampled_sequence_dto import SampledSequencesDTO
from reinvent_models.patformer.enums import SamplingModesEnum
from reinvent_models.patformer.models.vocabulary import SMILESTokenizer
from reinvent_models.patformer.patformer_model import PatformerModel


class PatformerAdapter(GenerativeModelBase):

    def __init__(self, path_to_file: str, mode: str):
        self.generative_model = PatformerModel.load_from_file(path_to_file, mode)
        self.vocabulary = self.generative_model.vocabulary
        self.max_sequence_length = self.generative_model.max_sequence_length
        self.network = self.generative_model.network
        self.tokenizer = SMILESTokenizer()

    def save_to_file(self, path):
        self.generative_model.save_to_file(path)

    def likelihood(self, src, src_mask, trg, trg_mask) -> torch.Tensor:
        return self.generative_model.likelihood(src, src_mask, trg, trg_mask)

    def likelihood_smiles(self, sampled_sequence_list: List[SampledSequencesDTO]) -> BatchLikelihoodDTO:
        input = [dto.input for dto in sampled_sequence_list]
        output = [dto.output for dto in sampled_sequence_list]
        dataset = PairedDataset(input, output, vocabulary=self.vocabulary, tokenizer=self.tokenizer)
        data_loader = tud.DataLoader(dataset, len(dataset), shuffle=False, collate_fn=PairedDataset.collate_fn)

        for batch in data_loader:
            likelihood = self.likelihood(batch.input, batch.input_mask, batch.output, batch.output_mask)
            dto = BatchLikelihoodDTO(batch, likelihood)
            return dto

    def sample(self, src, src_mask, decode_type=SamplingModesEnum.MULTINOMIAL) -> List[SampledSequencesDTO]:
        return self.generative_model.sample(src, src_mask, decode_type)

    def set_mode(self, mode: str):
        self.generative_model.set_mode(mode)

    def get_network_parameters(self):
        return self.generative_model.get_network_parameters()

    def get_vocabulary(self):
        return self.vocabulary
