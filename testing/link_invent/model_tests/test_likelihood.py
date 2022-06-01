import unittest

from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.link_invent_adapter import LinkInventAdapter
from reinvent_models.link_invent.dto.sampled_sequence_dto import SampledSequencesDTO
from testing.fixtures.paths import LINK_INVENT_PRIOR_PATH
from testing.fixtures.test_data import ETHANE, HEXANE, PROPANE, BUTANE


class TestLinkInventLikelihoodSMILES(unittest.TestCase):
    def setUp(self):
        dto1 = SampledSequencesDTO(ETHANE, PROPANE, 0.9)
        dto2 = SampledSequencesDTO(HEXANE, BUTANE, 0.1)
        self.sampled_sequence_list = [dto1, dto2]
        self._model = LinkInventAdapter(LINK_INVENT_PRIOR_PATH, mode=ModelModeEnum().INFERENCE)

    def test_len_likelihood_smiles(self):
        results = self._model.likelihood_smiles(self.sampled_sequence_list)
        self.assertEqual([2], list(results.likelihood.shape))