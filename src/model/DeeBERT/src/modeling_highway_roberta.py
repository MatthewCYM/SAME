from __future__ import absolute_import, division, print_function, unicode_literals

from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import RobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
)
import torch
from .modeling_highway_bert import BertPreTrainedModel, DeeBertModel, HighwayException, entropy
from .utils import batch_compute_early_exit_layer_and_prediction


class DeeRobertaModel(DeeBertModel):

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()


class DeeRobertaForSequenceClassification(BertPreTrainedModel):

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers

        self.roberta = DeeRobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_layer=-1,
        train_highway=False,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
                Classification (or regression if config.num_labels==1) loss.
            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
            highway_exits (:obj:`tuple(tuple(torch.Tensor))`:
                Tuple of each early exit's results (total length: number of layers)
                Each tuple is again, a tuple of length 2 - the first entry is logits and the second entry is hidden states.
        """

        exit_layer = self.num_layers
        try:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        except HighwayException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]

        if not self.training:
            original_entropy = entropy(logits)
            highway_entropy = []
            highway_logits_all = []
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # work with highway exits
            highway_losses = []
            for highway_exit in outputs[-1]:
                highway_logits = highway_exit[0]
                if not self.training:
                    highway_logits_all.append(highway_logits)
                    highway_entropy.append(highway_exit[2])
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    highway_loss = loss_fct(highway_logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    highway_loss = loss_fct(highway_logits.view(-1, self.num_labels), labels.view(-1))
                highway_losses.append(highway_loss)

            if train_highway:
                outputs = (sum(highway_losses[:-1]),) + outputs
                # exclude the final highway, of course
            else:
                outputs = (loss,) + outputs
        if not self.training:
            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
            if output_layer >= 0:
                outputs = (
                    (outputs[0],) + (highway_logits_all[output_layer],) + outputs[2:]
                )  # use the highway of the last layer

        return outputs  # (loss), logits, (hidden_states), (attentions), entropy

    def adv_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        early_exit_entropy=None,
        early_exit_patience=None,
        early_exit_reg_threshold=None
    ):
        assert self.training == False and early_exit_patience is None and early_exit_reg_threshold is None
        self.roberta.encoder.set_early_exit_entropy(-1)

        model_output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        highway_exits = model_output[1]
        highway_logits = [highway_exit[0] for highway_exit in highway_exits]
        highway_pooler_outputs = [highway_exit[1] for highway_exit in highway_exits]
        highway_entropies = [highway_exit[2] for highway_exit in highway_exits]
        highway_logits = torch.cat(highway_logits, dim=0)
        highway_pooler_outputs = torch.cat(highway_pooler_outputs, dim=0)
        highway_entropies = torch.stack(highway_entropies)
        exit_layer = -1
        for idx, item in enumerate(highway_entropies):
            if item < early_exit_entropy:
                exit_layer = idx
                break
        if exit_layer == -1:
            # no early exit
            prediction = torch.argmax(model_output[0])
            exit_layer = self.config.num_hidden_layers - 1
        else:
            # early exit
            prediction = torch.argmax(highway_logits[exit_layer])

        rtn = {
            'last_logits': model_output[0], # 1 * num_classes
            'highway_logits': highway_logits, # num_layers * num_classes
            'highway_pooler_outputs': highway_pooler_outputs,  # num_layers * hidden_dim
            'highway_entropies': highway_entropies, # num_layers * 1
            'exit_layer': exit_layer+1, # num computated layer
            'prediction': prediction, # final model prediction
        }
        return rtn

    def batch_adv_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        early_exit_entropy=None,
        early_exit_patience=None,
        early_exit_reg_threshold=None
    ):
        assert self.training == False and early_exit_patience is None and early_exit_reg_threshold is None
        self.roberta.encoder.set_early_exit_entropy(-1)

        model_outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        highway_exits = model_outputs[1]
        highway_logits = [highway_exit[0].unsqueeze(1) for highway_exit in highway_exits]
        highway_pooler_outputs = [highway_exit[1].unsqueeze(1) for highway_exit in highway_exits]
        highway_entropies = [highway_exit[2].unsqueeze(1) for highway_exit in highway_exits]
        highway_logits = torch.cat(highway_logits, dim=1)
        highway_pooler_outputs = torch.cat(highway_pooler_outputs, dim=1)
        highway_entropies = torch.cat(highway_entropies, dim=1)
        exit_layers, exit_logits, predictions = batch_compute_early_exit_layer_and_prediction(
            highway_entropies=highway_entropies,
            early_exit_entropy=early_exit_entropy,
            model_outputs=model_outputs,
            highway_logits=highway_logits,
            num_hidden_layers=self.config.num_hidden_layers
        )

        rtn = {
            # 'last_logits': model_outputs[0], # batch_size * num_classes
            'exit_logits': exit_logits, # batch_size * num_classes
            'highway_logits': highway_logits, # batch_size * num_layers * num_classes
            'highway_pooler_outputs': highway_pooler_outputs, # batch_size * num_layers * hidden_dim
            # 'highway_entropies': highway_entropies, # batch_size * num_layers
            'exit_layers': exit_layers, # batch_size
            'predictions': predictions, # batch_size
        }
        return rtn