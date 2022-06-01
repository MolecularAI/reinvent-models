import unittest

import torch.utils.data as tud

from reinvent_models.link_invent.dataset.dataset import Dataset
from reinvent_models.link_invent.link_invent_model import LinkInventModel

from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from testing.fixtures.paths import LINK_INVENT_PRIOR_PATH
from testing.fixtures.test_data import WARHEAD_PAIR


class TestLinkInventModel(unittest.TestCase):
    def setUp(self):

        self.smiles = WARHEAD_PAIR
        self._model = LinkInventModel.load_from_file(LINK_INVENT_PRIOR_PATH, mode=ModelModeEnum().INFERENCE)

        ds1 = Dataset([self.smiles], self._model.vocabulary.input)
        self.data_loader_1 = tud.DataLoader(ds1, batch_size=32, shuffle=False, collate_fn=Dataset.collate_fn)

        ds2 = Dataset([self.smiles] * 2, self._model.vocabulary.input)
        self.data_loader_2 = tud.DataLoader(ds2, batch_size=32, shuffle=False, collate_fn=Dataset.collate_fn)

        ds3 = Dataset([self.smiles] * 3, self._model.vocabulary.input)
        self.data_loader_3 = tud.DataLoader(ds3, batch_size=32, shuffle=False, collate_fn=Dataset.collate_fn)

    def _sample_linker(self, data_loader):
        results = []
        for warheads_batch in data_loader:
            for sampled_sequence in self._model.sample(*warheads_batch):
                results.append(sampled_sequence.output)
        return results

    def test_double_warheads_input(self):
        results = self._sample_linker(self.data_loader_2)
        self.assertEqual(2, len(results))

    def test_triple_warheads_input(self):
        results = self._sample_linker(self.data_loader_3)
        self.assertEqual(3, len(results))

    def test_single_warheads_input(self):
        results = self._sample_linker(self.data_loader_1)
        self.assertEqual(1, len(results))
