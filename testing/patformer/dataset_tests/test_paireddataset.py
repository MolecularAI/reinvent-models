import unittest

import torch
import torch.utils.data as tud

from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.patformer.dataset.paired_dataset import PairedDataset
from reinvent_models.patformer.models.vocabulary import SMILESTokenizer
from reinvent_models.patformer.patformer_model import PatformerModel
from testing.fixtures.paths import PATFORMER_PRIOR_PATH
from testing.fixtures.test_data import ETHANE, HEXANE, PROPANE, BUTANE


class TestPairedDataset(unittest.TestCase):
    def setUp(self):
        self.smiles_input = [ETHANE, PROPANE]
        self.smiles_output = [HEXANE, BUTANE]
        self._model = PatformerModel.load_from_file(PATFORMER_PRIOR_PATH, mode=ModelModeEnum().INFERENCE)
        self.data_loader = self.initialize_dataloader(self.smiles_input, self.smiles_output)

    def initialize_dataloader(self, smiles_input, smiles_output):
        dataset = PairedDataset(smiles_input, smiles_output, vocabulary=self._model.vocabulary, tokenizer=SMILESTokenizer())
        dataloader = tud.DataLoader(dataset, len(dataset), shuffle=False, collate_fn=PairedDataset.collate_fn)
        return dataloader

    def _get_src(self):
        for batch in self.data_loader:
            return batch.input

    def _get_src_mask(self):
        for batch in self.data_loader:
            return batch.input_mask

    def _get_trg(self):
        for batch in self.data_loader:
            return batch.output

    def _get_trg_mask(self):
        for batch in self.data_loader:
            return batch.output_mask

    def _get_src_shape(self):
        for batch in self.data_loader:
            return batch.input.shape

    def _get_src_mask_shape(self):
        for batch in self.data_loader:
            return batch.input_mask.shape

    def _get_trg_shape(self):
        for batch in self.data_loader:
            return batch.output.shape

    def _get_trg_mask_shape(self):
        for batch in self.data_loader:
            return batch.output_mask.shape

    def test_src_shape(self):
        result = self._get_src_shape()
        self.assertEqual(list(result), [2, 5])

    def test_src_mask_shape(self):
        result = self._get_src_mask_shape()
        self.assertEqual(list(result), [2, 1, 5])
    def test_trg_shape(self):
        result = self._get_trg_shape()
        self.assertEqual(list(result), [2, 8])

    def test_trg_mask_shape(self):
        result = self._get_trg_mask_shape()
        self.assertEqual(list(result), [2, 7, 7])

    def test_src_content(self):
        result = self._get_src().cpu()
        comparison = torch.equal(result, torch.tensor([[ 1, 17, 17,  2,  0],
                                                       [ 1, 17, 17, 17,  2]]))
        self.assertTrue(comparison)

    def test_src_mask_content(self):
        result = self._get_src_mask().cpu()
        comparison = torch.equal(result, torch.tensor([[[ True,  True,  True,  True, False]],
                                                       [[ True,  True,  True,  True,  True]]]))
        self.assertTrue(comparison)

    def test_trg_content(self):
        result = self._get_trg().cpu()
        comparison = torch.equal(result, torch.tensor([[ 1, 17, 17, 17, 17, 17, 17,  2],
                                                       [ 1, 17, 17, 17, 17,  2,  0,  0]]))
        self.assertTrue(comparison)

    def test_trg_mask_content(self):
        result = self._get_trg_mask().cpu()
        comparison = torch.equal(result, torch.tensor([
        [[ True, False, False, False, False, False, False],
         [ True,  True, False, False, False, False, False],
         [ True,  True,  True, False, False, False, False],
         [ True,  True,  True,  True, False, False, False],
         [ True,  True,  True,  True,  True, False, False],
         [ True,  True,  True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True,  True,  True]],

        [[ True, False, False, False, False, False, False],
         [ True,  True, False, False, False, False, False],
         [ True,  True,  True, False, False, False, False],
         [ True,  True,  True,  True, False, False, False],
         [ True,  True,  True,  True,  True, False, False],
         [ True,  True,  True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True,  True, False]]]))
        self.assertTrue(comparison)

