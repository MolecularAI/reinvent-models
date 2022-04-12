import unittest

import torch.utils.data as tud

from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.patformer.dataset.dataset import Dataset
from reinvent_models.patformer.enums import SamplingModesEnum
from reinvent_models.patformer.models.vocabulary import SMILESTokenizer
from reinvent_models.patformer.patformer_model import PatformerModel
from testing.fixtures.paths import PATFORMER_PRIOR_PATH
from testing.fixtures.test_data import METAMIZOLE, COCAINE, AMOXAPINE


class TestModelSampling(unittest.TestCase):
    def setUp(self):

        self._model = PatformerModel.load_from_file(PATFORMER_PRIOR_PATH, mode=ModelModeEnum().INFERENCE)
        self._sample_mode_enum = SamplingModesEnum()

        smiles_list = [METAMIZOLE]
        self.data_loader_1 = self.initialize_dataloader(smiles_list)

        smiles_list = [METAMIZOLE, COCAINE]
        self.data_loader_2 = self.initialize_dataloader(smiles_list)

        smiles_list = [METAMIZOLE, COCAINE, AMOXAPINE]
        self.data_loader_3 = self.initialize_dataloader(smiles_list)

    def initialize_dataloader(self, data):
        dataset = Dataset(data, vocabulary=self._model.vocabulary, tokenizer=SMILESTokenizer())
        dataloader = tud.DataLoader(dataset, len(dataset), shuffle=False, collate_fn=Dataset.collate_fn)
        return dataloader

    def _sample_molecules(self, data_loader):
        results = []
        for batch in data_loader:
            src, src_mask  = batch
            for sampled_sequence in self._model.sample(src, src_mask, decode_type=self._sample_mode_enum.MULTINOMIAL):
                results.append(sampled_sequence.output)
        return results

    def test_single_input(self):
        results = self._sample_molecules(self.data_loader_1)
        self.assertEqual(1, len(results))

    def test_double_input(self):
        results = self._sample_molecules(self.data_loader_2)
        self.assertEqual(2, len(results))

    def test_triple_input(self):
        results = self._sample_molecules(self.data_loader_3)
        self.assertEqual(3, len(results))


