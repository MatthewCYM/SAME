import numpy as np
import torch.nn.functional as F
import transformers
import torch
from textattack.models.wrappers import ModelWrapper
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def soft_cross_entropy(preds, labels):
    pred_likelihoods = F.log_softmax(preds, dim=-1)
    return - labels * pred_likelihoods


class DynamicTransformerClassifier(ModelWrapper):
    def __init__(self,
                 model: transformers.PreTrainedModel,
                 tokenizer: transformers.PreTrainedTokenizer,
                 device,
                 args=None,
                 ):
        """
        Args:
            model: Huggingface model for classification.
            tokenizer: Huggingface tokenizer for classification. **Default:** None
            device: Device of pytorch model. **Default:** "cpu" if cuda is not available else "cuda"
        """
        self.args = args
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.max_length = args.max_seq_length
        self.device = device
        self.early_exit_entropy = args.early_exit_entropy
        self.early_exit_patience = args.early_exit_patience
        self.loss_func = torch.nn.CrossEntropyLoss()

        self.lam = self.args.lam
        self.model_type = self.args.model_type

        self.num_exits = self.model.config.num_hidden_layers

        if args.model_type == 'deebert':
            self.embedding = self.model.bert.embeddings.word_embeddings.weight
        elif args.model_type == 'deeroberta':
            self.embedding = self.model.roberta.embeddings.word_embeddings.weight
        elif args.model_type == 'pabeebert':
            self.embedding = self.model.bert.embeddings.word_embeddings.weight
        elif args.model_type == 'pabeealbert':
            self.embedding = self.model.albert.embeddings.word_embeddings.weight
        else:
            raise NotImplementedError

    def _tokenize(self, inputs: List[Tuple[str, str]]):
        text_a = [x[0] for x in inputs]
        text_b = [x[1] if x[1] is not '' else None for x in inputs]
        if None in text_b:
            new_inputs = self.tokenizer(
                text=text_a,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True
            ).data
        else:
            new_inputs = self.tokenizer(
                text=text_a,
                text_pair=text_b,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True
            ).data
        new_inputs = {k: new_inputs[k].to(self.device) for k in new_inputs}
        return new_inputs

    def _tokenize_string(self, text_input_list: List[str]):
        new_inputs = self.tokenizer(
            text=text_input_list,
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True
        ).data
        new_inputs = {k: new_inputs[k].to(self.device) for k in new_inputs}
        return new_inputs

    def __call__(self, text_input_list: List[Tuple[str, str]]):
        assert isinstance(text_input_list[0], tuple)
        assert isinstance(text_input_list[0][0], str)

        rnt = self.prediction(text_input_list)
        raw_scores = rnt['scores'].detach().cpu()
        return raw_scores

    def get_grad(self, text_input):
        raise NotImplementedError

    def get_weighted_uniform_target_ce_loss(self, rtn, weight):
        highway_logits = rtn['highway_logits']
        highway_logits = highway_logits.squeeze(0)
        uniform_target = torch.ones_like(highway_logits) / self.model.num_labels
        loss = soft_cross_entropy(highway_logits, uniform_target).mean(dim=-1)
        loss = loss * weight
        loss = loss.mean()
        return loss

    def get_weighted_heuristic_cross_prediction_loss(self, rtn, weight):
        highway_logits = rtn['highway_logits']
        highway_logits_permuted = highway_logits.squeeze(0)

        softmax_value = torch.nn.Softmax(dim=2)(highway_logits)
        prediction = (-softmax_value).argsort(2)[0, :, 0]
        second_prediction = (-softmax_value).argsort(2)[0, :, 1]

        prev_pred = int(prediction[0])
        heuristic_target = [prev_pred]
        for i in range(1, highway_logits_permuted.shape[0]):
            current_pred = int(prediction[i])
            if current_pred == prev_pred:
                heuristic_target.append(second_prediction[i])
            else:
                heuristic_target.append(prediction[i])
            prev_pred = current_pred
        heuristic_target = torch.tensor(heuristic_target, device=self.device, dtype=torch.long)
        loss = torch.nn.CrossEntropyLoss(reduction='none')(highway_logits_permuted, heuristic_target) * weight
        loss = loss.mean()
        return loss

    def get_slow_grad(self, text_input):
        rtn = self.prediction([text_input])
        weight = torch.FloatTensor([0.1 if i < rtn['exit_layers'] - 1 else 1.2 ** (i - rtn['exit_layers'] + 1) for i in range(self.num_exits)])
        weight = weight.to(rtn['highway_logits'].device)

        if self.lam != 0:
            loss1 = self.get_weighted_heuristic_cross_prediction_loss(rtn, weight)
        else:
            loss1 = 0
        if self.lam != 1:
            loss2 = self.get_weighted_uniform_target_ce_loss(rtn, weight)
        else:
            loss2 = 0.0
        loss = self.lam * loss1 + (1 - self.lam) * loss2
        loss.backward()
        grad = self.embedding.grad.detach().cpu().numpy()
        rtn['gradient'] = grad
        return rtn

    def get_pred(self, text_input_list: List[str] or List[Tuple[str, str]]):
        rnt = self.prediction(text_input_list)
        return rnt['predictions']

    def prediction(self, text_input_list: List[str] or List[Tuple[str, str]]):
        assert isinstance(text_input_list[0], tuple)
        assert isinstance(text_input_list[0][0], str)

        if isinstance(text_input_list[0], tuple):
            input_tensor = self._tokenize(text_input_list)
        elif type(text_input_list[0]) is str:
            input_tensor = self._tokenize_string(text_input_list)
        else:
            raise NotImplementedError
        rnt = self.model.batch_adv_forward(
            **input_tensor, early_exit_entropy=self.early_exit_entropy, early_exit_patience=self.early_exit_patience)

        logits = rnt['exit_logits']
        text_ids = input_tensor['input_ids']

        norm_func = torch.nn.Softmax(dim=-1)
        scores = norm_func(logits)
        highway_scores = norm_func(rnt['highway_logits'])

        processed_rnt = {
            'logits': logits,
            'exit_layers': rnt['exit_layers'],
            'predictions': rnt['predictions'],
            'highway_logits': rnt['highway_logits'],
            'highway_scores': highway_scores,
            'highway_pooler_outputs': rnt['highway_pooler_outputs'],
            'scores': scores,
            'ids': text_ids
        }
        return processed_rnt
