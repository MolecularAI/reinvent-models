from typing import Union, Any

import torch
from dacite import from_dict
from torch import nn as tnn
from torch.autograd import Variable

from reinvent_models.model_factory.enums.model_mode_enum import ModelModeEnum
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.patformer.dto import PatformerModelParameterDTO
from reinvent_models.patformer.dto.sampled_sequence_dto import SampledSequencesDTO
from reinvent_models.patformer.enums import SamplingModesEnum
from reinvent_models.patformer.models.encode_decode.model import EncoderDecoder
from reinvent_models.patformer.models.module.subsequent_mask import subsequent_mask
from reinvent_models.patformer.models.vocabulary import SMILESTokenizer
from reinvent_models.patformer.models.vocabulary import Vocabulary


class PatformerModel():
    def __init__(self, vocabulary: Vocabulary, network: EncoderDecoder,
                 max_sequence_length: int = 128, no_cuda: bool = False, mode: str = ModelModeEnum().TRAINING):
        self.vocabulary = vocabulary
        self.tokenizer =  SMILESTokenizer()
        self.network = network
        self.max_sequence_length = max_sequence_length

        self._model_modes = ModelModeEnum()
        self._sampling_modes_enum = SamplingModesEnum()

        self.set_mode(mode)
        if torch.cuda.is_available() and not no_cuda:
            self.network.cuda()
        self._nll_loss = tnn.NLLLoss(reduction="none", ignore_index=0)

    def set_mode(self, mode: str):
        if mode == self._model_modes.TRAINING:
            self.network.train()
        elif mode == self._model_modes.INFERENCE:
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")

    @classmethod
    def load_from_file(cls, path_to_file, mode: str = ModelModeEnum().TRAINING) -> Union[Any, GenerativeModelBase] :
        """
        Loads a model from a single file
        :param path_to_file: Path to the saved model
        :param mode: Mode in which the model should be initialized
        :return: An instance of the network
        """
        data = from_dict(PatformerModelParameterDTO, torch.load(path_to_file))
        network = EncoderDecoder(**data.network_parameter)
        network.load_state_dict(data.network_state)
        model = PatformerModel(vocabulary=data.vocabulary, network=network,
                                max_sequence_length=data.max_sequence_length, mode=mode)
        return model

    def save_to_file(self, path_to_file):
        """
        Saves the model to a file.
        :param path_to_file: Path to the file which the model will be saved to.
        """
        data = PatformerModelParameterDTO(vocabulary=self.vocabulary, max_sequence_length=self.max_sequence_length,
                                           network_parameter=self.network.get_params(),
                                           network_state=self.network.state_dict())
        torch.save(data.__dict__, path_to_file)

    def likelihood(self, src, src_mask, trg, trg_mask):
        """
        Retrieves the likelihood of molecules.
        :param src: (batch, seq) A batch of padded input sequences.
        :param src_mask: (batch, seq, seq) Mask of the input sequences.
        :param trg: (batch, seq) A batch of output sequences; with start token, without end token.
        :param trg_mask: Mask of the input sequences.
        :return:  (batch) Log likelihood for each output sequence in the batch.
        """
        trg_y = trg[:, 1:] # skip start token but keep end token
        trg = trg[:, :-1] # save start token, skip end token
        out = self.network.forward(src, trg, src_mask, trg_mask)
        log_prob = self.network.generator(out).transpose(1, 2) #(batch, voc, seq_len)
        nll = self._nll_loss(log_prob, trg_y).sum(dim=1)

        return nll

    @torch.no_grad()
    def sample(self, src, src_mask, decode_type):
        batch_size = src.shape[0]
        ys = torch.ones(1)
        ys = ys.repeat(batch_size, 1).view(batch_size, 1).type_as(src.data) # shape [batch_size, 1]
        encoder_outputs = self.network.encode(src, src_mask)
        break_condition = torch.zeros(batch_size, dtype=torch.bool)

        nlls = torch.zeros(src.shape[0]).cuda()
        loss = torch.nn.NLLLoss(reduction="none", ignore_index=0)  # note 0 is padding
        for i in range(self.max_sequence_length - 1):
            out = self.network.decode(encoder_outputs, src_mask, Variable(ys),
                               Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
            # (batch, seq, voc) need to exclude the probability of the start token "1"
            log_prob = self.network.generator(out[:, -1])
            prob = torch.exp(log_prob)

            if decode_type == self._sampling_modes_enum.GREEDY:
                _, next_word = torch.max(prob, dim=1)
                # mask numbers after end token as 0
                next_word = next_word.masked_fill(break_condition.cuda(), 0)
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)  # [batch_size, i]

                # This does the same as loss(), logprob = torch.index_select(log_prob, 1, next_word)
                nlls += loss(log_prob, next_word)
            elif decode_type == self._sampling_modes_enum.MULTINOMIAL:
                next_word = torch.multinomial(prob, 1)
                # mask numbers after end token as 0
                break_t = torch.unsqueeze(break_condition, 1).cuda()
                next_word = next_word.masked_fill(break_t, 0)
                ys = torch.cat([ys, next_word], dim=1)  # [batch_size, i]
                next_word = torch.reshape(next_word, (next_word.shape[0],))

                # This does the same as loss(), logprob = torch.index_select(log_prob, 1, next_word)
                nlls += loss(log_prob, next_word)

            # next_word = np.array(next_word.to('cpu').tolist())
            break_condition = (break_condition | (next_word.to('cpu') == 2))
            if all(break_condition):  # end token
                break

        tokenizer = SMILESTokenizer()
        output_smiles_list = [tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in ys.data.cpu().numpy()]
        input_smiles_list = [tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in src.data.cpu().numpy()]
        result = [SampledSequencesDTO(input, output, nll) for input, output, nll in
                  zip(input_smiles_list, output_smiles_list, nlls.data.cpu().numpy().tolist())]

        return result

    def get_network_parameters(self):
        return self.network.parameters()