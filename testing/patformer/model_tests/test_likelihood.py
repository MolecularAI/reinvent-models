import unittest

import torch.utils.data as tud

from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.patformer.dataset.paired_dataset import PairedDataset
from reinvent_models.patformer.models.vocabulary import SMILESTokenizer
from reinvent_models.patformer.patformer_model import PatformerModel
from testing.fixtures.paths import PATFORMER_PRIOR_PATH
from testing.fixtures.test_data import ETHANE, HEXANE, PROPANE, BUTANE


class TestLikelihood(unittest.TestCase):
    def setUp(self) -> None:
        self.smiles_input = [ETHANE, PROPANE]
        self.smiles_output = [HEXANE, BUTANE]

        self._model = PatformerModel.load_from_file(PATFORMER_PRIOR_PATH, mode=ModelModeEnum().INFERENCE)
        self.data_loader = self.initialize_dataloader(self.smiles_input, self.smiles_output)

    def initialize_dataloader(self, input, output):
        dataset = PairedDataset(input, output, vocabulary=self._model.vocabulary, tokenizer=SMILESTokenizer())
        dataloader = tud.DataLoader(dataset, len(dataset), shuffle=False, collate_fn=PairedDataset.collate_fn)
        return dataloader

    def test_len_likelihood(self):
        for batch in self.data_loader:
            results = self._model.likelihood(batch.input, batch.input_mask, batch.output, batch.output_mask)
            self.assertEqual([2], list(results.shape))


