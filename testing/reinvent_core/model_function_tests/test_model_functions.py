import os
import shutil
import unittest

import numpy
import numpy.testing as nt
import torch

from reinvent_models.reinvent_core.models.model import Model
from testing.fixtures.paths import MAIN_TEST_PATH, PRIOR_PATH
from testing.fixtures.test_data import PROPANE, BENZENE, METAMIZOLE
from testing.utils.general import set_default_device_cuda


class TestModelFunctions(unittest.TestCase):

    def setUp(self):
        set_default_device_cuda()
        self.workfolder = MAIN_TEST_PATH
        self.output_file = os.path.join(self.workfolder, "generative_model.ckpt")
        if not os.path.isdir(self.workfolder):
            os.makedirs(self.workfolder)
        self.model = Model.load_from_file(PRIOR_PATH)

    def tearDown(self):
        if os.path.isdir(self.workfolder):
            shutil.rmtree(self.workfolder)

        set_default_device_cuda(dont_use_cuda=True)

    def test_likelihoods_from_model_1(self):
        likelihoods = self.model.likelihood_smiles([PROPANE, BENZENE])
        self.assertAlmostEqual(likelihoods[0].item(), 20.9116, 3)
        self.assertAlmostEqual(likelihoods[1].item(), 17.9506, 3)
        self.assertEqual(len(likelihoods), 2)
        self.assertEqual(type(likelihoods), torch.Tensor)

    def test_likelihoods_from_model_2(self):
        likelihoods = self.model.likelihood_smiles(
            [METAMIZOLE])
        self.assertAlmostEqual(likelihoods[0].item(), 125.4669, 3)
        self.assertEqual(len(likelihoods), 1)
        self.assertEqual(type(likelihoods), torch.Tensor)

    def test_sample_from_model_1(self):
        sample, nll = self.model.sample_smiles(num=20, batch_size=20)
        self.assertEqual(len(sample), 20)
        self.assertEqual(type(sample), list)
        self.assertEqual(len(nll), 20)
        self.assertEqual(type(nll), numpy.ndarray)

    def test_sample_from_model_2(self):
        seq, sample, nll = self.model.sample_sequences_and_smiles(batch_size=20)
        self.assertEqual(seq.shape[0], 20)
        self.assertEqual(type(seq), torch.Tensor)
        self.assertEqual(len(sample), 20)
        self.assertEqual(type(sample), list)
        self.assertEqual(len(nll), 20)
        self.assertEqual(type(nll), torch.Tensor)

    def test_model_tokens(self):
        tokens = self.model.vocabulary.tokens()
        self.assertIn('C', tokens)
        self.assertIn('O', tokens)
        self.assertIn('Cl', tokens)
        self.assertEqual(len(tokens), 34)
        self.assertEqual(type(tokens), list)

    def test_save_model(self):
        self.model.save(self.output_file)
        self.assertEqual(os.path.isfile(self.output_file), True)

    def test_likelihood_function_differences(self):
        seq, sample, nll = self.model.sample_sequences_and_smiles(batch_size=128)
        nll2 = self.model.likelihood(seq)
        nll3 = self.model.likelihood_smiles(sample)
        nt.assert_array_almost_equal(nll.detach().cpu().numpy(), nll2.detach().cpu().numpy(), 3)
        nt.assert_array_almost_equal(nll.detach().cpu().numpy(), nll3.detach().cpu().numpy(), 3)