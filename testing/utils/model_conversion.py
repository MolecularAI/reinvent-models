import copy
import unittest

import numpy as np
import torch
import torch.utils.data as tud

import reinvent_models.reinvent_core.models.model as new_model
from reinvent_models.reinvent_core.models.dataset import calculate_nlls_from_model, Dataset
# from models.model import Model
from reinvent_models.reinvent_core.models.vocabulary import SMILESTokenizer, Vocabulary
from testing.fixtures.paths import OLD_PRIOR_PATH, PRIOR_PATH
from testing.fixtures.test_data import SIMPLE_TOKENS, ASPIRIN, CELECOXIB, METAMIZOLE
from testing.utils.general import set_default_device_cuda


class Test_dataset_functions(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        set_default_device_cuda()
        self.smiles_list = [ASPIRIN, CELECOXIB, METAMIZOLE]
        self.Dataset = Dataset(self.smiles_list, SIMPLE_TOKENS, SMILESTokenizer)
        self.coldata = tud.DataLoader(self.smiles_list, 1, collate_fn=Dataset.collate_fn)

    def tearDown(self):
        set_default_device_cuda(dont_use_cuda=True)

    def test_dataset_functions_1(self):
        self.assertEqual(len(self.coldata), 3)

    def test_dataset_functions_2(self):
        # PRIOR_PATH = f'{PRIOR_PATH}.new'
        file = new_model.Model.load_from_file(PRIOR_PATH, sampling_mode=True)
        nll_calc, nllcalc2 = calculate_nlls_from_model(file, self.smiles_list, batch_size=2)
        nllmat = np.concatenate(list(nll_calc))
        self.assertEqual(len(nllmat), 3)
        self.assertAlmostEqual(nllmat[0], 17.568218, 3)
        self.assertAlmostEqual(nllmat[1], 257.06732, 3)

    def convert_old_model_to_new(self): #needs test_ prefix

        original = torch.load(OLD_PRIOR_PATH)
        # original = Model.load_from_file(PRIOR_PATH, sampling_mode=True)
        # save_dict = {
        #     'vocabulary': original.vocabulary,
        #     'tokenizer': original.tokenizer,
        #     'max_sequence_length': original.max_sequence_length,
        #     'network': original.network.state_dict(),
        #     'network_params': original.network.get_params()
        # }
        original_vocabulary = original['vocabulary']
        tokens = {i:v for i, v in original_vocabulary._tokens.items()}
        vocabulary = Vocabulary()
        vocabulary._tokens = copy.copy(original_vocabulary._tokens)
        vocabulary._current_id = 37
        # for k, v in original_vocabulary._tokens.items():
        #     vocabulary.add(v)
        network_params = original.get("network_params", {})
        model = new_model.Model(
            vocabulary=vocabulary,
            tokenizer=SMILESTokenizer(),
            network_params=network_params,
            max_sequence_length=original['max_sequence_length']
        )
        model.network.load_state_dict(original["network"])
        model.save(PRIOR_PATH)

