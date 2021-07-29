import copy
import unittest

import torch

import reinvent_models.lib_invent.models.decorator as new_mdec
import reinvent_models.lib_invent.models.model as new_mm
import reinvent_models.lib_invent.models.vocabulary as new_voc
from testing.fixtures.paths import OLD_LIBINVENT_PRIOR_PATH, LIBINVENT_PRIOR_PATH


# from models.model import Model
# import models.model as mm
# import models.decorator as mdec

class ConvertLibInventModel(unittest.TestCase):
    def setUp(self):
        self.smiles = "c1ccccc1CC0C"
        self.actor = new_mm.DecoratorModel.load_from_file(LIBINVENT_PRIOR_PATH, mode="train")

    def test_something(self):
        tokenized = self.actor.vocabulary.scaffold_tokenizer.tokenize(self.smiles)
        # encoded = self.actor.vocabulary.encode(tokenized)
        self.assertEqual(14, len(tokenized))

    def convert_old_model_to_new(self): #needs test_ prefix
        data = torch.load(OLD_LIBINVENT_PRIOR_PATH)

        dec_vocabulary: new_voc.DecoratorVocabulary = data['model']['vocabulary']
        dec_vocabulary.decoration_tokenizer = new_voc.SMILESTokenizer()
        dec_vocabulary.scaffold_tokenizer = new_voc.SMILESTokenizer()

        new_dec_vocabulary = new_voc.Vocabulary()
        new_dec_vocabulary._tokens = copy.copy(dec_vocabulary.decoration_vocabulary._tokens)
        new_dec_vocabulary._current_id = 36
        dec_vocabulary.decoration_vocabulary = new_dec_vocabulary

        ############################

        new_scaff_vocabulary = new_voc.Vocabulary()
        new_scaff_vocabulary._tokens = copy.copy(dec_vocabulary.scaffold_vocabulary._tokens)
        new_scaff_vocabulary._current_id = 38
        dec_vocabulary.scaffold_vocabulary = new_scaff_vocabulary

        ##############################

        decorator_vocabulary = new_voc.DecoratorVocabulary(dec_vocabulary.scaffold_vocabulary,
                                                           dec_vocabulary.scaffold_tokenizer,
                                                           dec_vocabulary.decoration_vocabulary,
                                                           dec_vocabulary.decoration_tokenizer)
        data['model']['vocabulary'] = decorator_vocabulary

        decorator = new_mdec.Decorator(**data["decorator"]["params"])
        decorator.load_state_dict(data["decorator"]["state"])


        model = new_mm.DecoratorModel(
            decorator=decorator,
            mode="train",
            **data["model"]
        )

        model.save(LIBINVENT_PRIOR_PATH)