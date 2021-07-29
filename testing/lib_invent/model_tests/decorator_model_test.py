import unittest

import torch.utils.data as tud
from reinvent_models.lib_invent.models.dataset import Dataset

from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.lib_invent.models.model import DecoratorModel
from testing.fixtures.paths import LIBINVENT_PRIOR_PATH
from testing.fixtures.test_data import SCAFFOLD_SUZUKI


class TestDecoratorModel(unittest.TestCase):
    def setUp(self):
        input_scaffold = SCAFFOLD_SUZUKI
        scaffold_list_2 = [input_scaffold, input_scaffold]
        scaffold_list_3 = [input_scaffold, input_scaffold, input_scaffold]
        self._model_regime = GenerativeModelRegimeEnum()
        self._decorator = DecoratorModel.load_from_file(LIBINVENT_PRIOR_PATH, mode=self._model_regime.INFERENCE)

        dataset_2 = Dataset(scaffold_list_2, self._decorator.vocabulary.scaffold_vocabulary,
                               self._decorator.vocabulary.scaffold_tokenizer)
        self.dataloader_2 = tud.DataLoader(dataset_2, batch_size=32, shuffle=False, collate_fn=Dataset.collate_fn)

        dataset_3 = Dataset(scaffold_list_3, self._decorator.vocabulary.scaffold_vocabulary,
                               self._decorator.vocabulary.scaffold_tokenizer)
        self.dataloader_3 = tud.DataLoader(dataset_3, batch_size=32, shuffle=False, collate_fn=Dataset.collate_fn)

    def test_double_scaffold_input(self):
        results = []
        for batch in self.dataloader_2:
            scaffold_seqs, scaffold_seq_lengths = batch

            for scaff, decorations, nll in self._decorator.sample_decorations(scaffold_seqs, scaffold_seq_lengths):
                results.append(decorations)
        self.assertEqual(2, len(results))

    def test_triple_scaffold_input(self):
        results = []
        for batch in self.dataloader_3:
            for scaff, decorations, nll in self._decorator.sample_decorations(*batch):
                results.append(decorations)
        self.assertEqual(3, len(results))
