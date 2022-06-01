import unittest

from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.lib_invent_adapter import LibInventAdapter
from reinvent_models.link_invent.dto.sampled_sequence_dto import SampledSequencesDTO
from testing.fixtures.paths import LIBINVENT_PRIOR_PATH
from testing.fixtures.test_data import ETHANE, HEXANE, PROPANE, BUTANE


class TestLibInventLikelihoodSMILES(unittest.TestCase):
    def setUp(self):
        dto1 = SampledSequencesDTO(ETHANE, PROPANE, 0.9)
        dto2 = SampledSequencesDTO(HEXANE, BUTANE, 0.1)
        self.sampled_sequence_list = [dto1, dto2]
        self._model = LibInventAdapter(LIBINVENT_PRIOR_PATH, mode=ModelModeEnum().INFERENCE)

    def test_len_likelihood_smiles(self):
        results = self._model.likelihood_smiles(self.sampled_sequence_list)
        self.assertEqual([2], list(results.likelihood.shape))