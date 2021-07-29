import unittest

from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.lib_invent.models.model import DecoratorModel
from testing.fixtures.paths import LIBINVENT_PRIOR_PATH


class TestTokenizationWithModel(unittest.TestCase):
    def setUp(self):
        self._model_regime = GenerativeModelRegimeEnum()
        self.smiles = "c1ccccc1CC0C"
        self.actor = DecoratorModel.load_from_file(LIBINVENT_PRIOR_PATH, mode=self._model_regime.TRAINING)

    def test_tokenization(self):
        tokenized = self.actor.vocabulary.scaffold_tokenizer.tokenize(self.smiles)
        self.assertEqual(14, len(tokenized))
