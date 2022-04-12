import unittest

import numpy as np
import numpy.testing as npt

import torch.utils.data as tud

from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.patformer_adapter import PatformerAdapter
from reinvent_models.patformer.dataset.dataset import Dataset
from reinvent_models.patformer.enums import SamplingModesEnum
from reinvent_models.patformer.models.vocabulary import SMILESTokenizer
from testing.fixtures.paths import PATFORMER_PRIOR_PATH
from testing.fixtures.test_data import METAMIZOLE, COCAINE, AMOXAPINE


class TestSampleLikelihoodSMILES(unittest.TestCase):
    def setUp(self):

        self._model = PatformerAdapter(PATFORMER_PRIOR_PATH, mode=ModelModeEnum().INFERENCE)
        self._sample_mode_enum = SamplingModesEnum()

        smiles_list = [METAMIZOLE, COCAINE, AMOXAPINE]
        self.data_loader = self.initialize_dataloader(smiles_list)

    def initialize_dataloader(self, data):
        dataset = Dataset(data, vocabulary=self._model.vocabulary, tokenizer=SMILESTokenizer())
        dataloader = tud.DataLoader(dataset, len(dataset), shuffle=False, collate_fn=Dataset.collate_fn)
        return dataloader

    def _sample_molecules(self, data_loader):
        results = []
        for batch in data_loader:
            src, src_mask = batch
            for sampled_sequence in self._model.sample(src, src_mask,
                                                       decode_type=self._sample_mode_enum.MULTINOMIAL):
                results.append(sampled_sequence)
        return results

    def test_sample_likelihood_smiles_consistency(self):
        sampled_sequence_list = self._sample_molecules(self.data_loader)

        sampled_nlls_list = []
        for sampled_sequence_dto in sampled_sequence_list:
            sampled_nlls_list.append(sampled_sequence_dto.nll)
        sampled_nlls_array = np.array(sampled_nlls_list)

        batch_likelihood_dto = self._model.likelihood_smiles(sampled_sequence_list)
        likelihood_smiles_nlls_array = batch_likelihood_dto.likelihood.cpu().detach().numpy()

        npt.assert_array_almost_equal(sampled_nlls_array, likelihood_smiles_nlls_array, decimal=4)
