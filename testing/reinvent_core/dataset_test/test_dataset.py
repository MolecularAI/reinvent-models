import unittest

import numpy as np
import torch.utils.data as tud

import reinvent_models.reinvent_core.models.model as new_model
from reinvent_models.reinvent_core.models.dataset import calculate_nlls_from_model, Dataset
from reinvent_models.reinvent_core.models.vocabulary import SMILESTokenizer
from testing.fixtures.paths import PRIOR_PATH
from testing.fixtures.test_data import SIMPLE_TOKENS, HEXANE, PROPANE, ETHANE, BUTANE, ASPIRIN
from testing.utils.general import set_default_device_cuda


class TestDatasetFunctions(unittest.TestCase):

    def setUp(self):
        set_default_device_cuda()
        self.smiles = [ASPIRIN, BUTANE, ETHANE, PROPANE, HEXANE]
        self.Dataset = Dataset(self.smiles, SIMPLE_TOKENS, SMILESTokenizer)
        self.coldata = tud.DataLoader(self.smiles, 1, collate_fn=Dataset.collate_fn)

    def tearDown(self):
        set_default_device_cuda(dont_use_cuda=True)

    def test_dataset_functions_1(self):
        self.assertEqual(len(self.coldata), 5)

    def test_dataset_functions_2(self):
        file = new_model.Model.load_from_file(PRIOR_PATH, sampling_mode=True)
        nll_calc, _ = calculate_nlls_from_model(file, self.smiles, batch_size=5)
        nllmat = np.concatenate(list(nll_calc))
        self.assertEqual(len(nllmat), 5)
        self.assertAlmostEqual(nllmat[0], 17.568214, 3)
        self.assertAlmostEqual(nllmat[1], 18.699503, 3)