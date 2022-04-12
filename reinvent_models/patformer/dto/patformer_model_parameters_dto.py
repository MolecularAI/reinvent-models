from dataclasses import dataclass

from reinvent_models.patformer.models.vocabulary import Vocabulary

@dataclass
class PatformerModelParameterDTO:
    vocabulary: Vocabulary
    max_sequence_length: int
    network_parameter: dict
    network_state: dict