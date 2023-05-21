import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.albert.configuration_albert import AlbertConfig
from transformers.models.albert.modeling_albert import (
    load_tf_weights_in_albert,
    AlbertEmbeddings,
    AlbertAttention,
    AlbertLayer,
    AlbertPreTrainedModel,
    AlbertLayerGroup
)
from .utils import batch_compute_early_exit_layer_and_prediction


logger = logging.getLogger(__name__)


ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "albert-base-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-pytorch_model.bin",
    "albert-large-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-pytorch_model.bin",
    "albert-xlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-pytorch_model.bin",
    "albert-xxlarge-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-pytorch_model.bin",
    "albert-base-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-pytorch_model.bin",
    "albert-large-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-pytorch_model.bin",
    "albert-xlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v2-pytorch_model.bin",
    "albert-xxlarge-v2": "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-pytorch_model.bin",
}


class AlbertTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_attentions = ()

        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
            )
            hidden_states = layer_group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None):
        if current_layer == 0:
            hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        else:
            hidden_states = hidden_states[0]

        layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

        # Index of the hidden group
        group_idx = int(current_layer / (self.config.num_hidden_layers / self.config.num_hidden_groups))

        # Index of the layer inside the group
        layer_idx = int(current_layer - group_idx * layers_per_group)

        layer_group_output = self.albert_layer_groups[group_idx](hidden_states, attention_mask, head_mask[group_idx * layers_per_group:(group_idx + 1) * layers_per_group])
        hidden_states = layer_group_output[0]

        return (hidden_states,)


class AlbertModel(AlbertPreTrainedModel):

    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "albert"

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.init_weights()
        self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0

        self.regression_threshold = 0

    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold

    def set_patience(self, patience):
        self.patience = patience

    def reset_stats(self):
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self):
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = f'*** Patience = {self.patience} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up = {1 - avg_inf_layers / self.config.num_hidden_layers:.2f} ***'
        print(message)
        rtn = {
            'avg_inf_layer': avg_inf_layers,
            'speed ratio': self.config.num_hidden_layers / avg_inf_layers
        }
        return rtn

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_dropout=None,
        output_layers=None,
        regression=False
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = embedding_output

        if self.training:
            res = []
            for i in range(self.config.num_hidden_layers):
                encoder_outputs = self.encoder.adaptive_forward(encoder_outputs,
                                                                current_layer=i,
                                                                attention_mask=extended_attention_mask,
                                                                head_mask=head_mask
                                                                )

                pooled_output = self.pooler_activation(self.pooler(encoder_outputs[0][:, 0]))
                logits = output_layers[i](output_dropout(pooled_output))
                res.append(logits)
        elif self.patience == 0:  # Use all layers for inference
            encoder_outputs = self.encoder(encoder_outputs,
                                           extended_attention_mask,
                                           head_mask=head_mask)
            pooled_output = self.pooler_activation(self.pooler(encoder_outputs[0][:, 0]))
            res = [output_layers[self.config.num_hidden_layers - 1](pooled_output)]
        elif self.patience == -666:
            highway_logits = []
            highway_pooler_outputs = []
            for i in range(self.config.num_hidden_layers):
                encoder_outputs = self.encoder.adaptive_forward(encoder_outputs,
                                                                current_layer=i,
                                                                attention_mask=extended_attention_mask,
                                                                head_mask=head_mask)
                pooled_output = self.pooler_activation(self.pooler(encoder_outputs[0][:, 0]))
                logits = output_layers[i](pooled_output)
                highway_logits.append(logits) # num_layer * bs * num_class
                highway_pooler_outputs.append(pooled_output) # num_layer * bs * dim

            highway_logits = torch.stack(highway_logits).permute(1, 0, 2)  # bs * num_layer * num_class
            highway_pooler_outputs = torch.stack(highway_pooler_outputs).permute(1, 0, 2)  # bs * num_layer * num_class

            res = {
                'highway_logits': highway_logits,
                'highway_pooler_outputs': highway_pooler_outputs
            }
        else:
            patient_counter = 0
            patient_result = None
            calculated_layer_num = 0
            for i in range(self.config.num_hidden_layers):
                calculated_layer_num += 1
                encoder_outputs = self.encoder.adaptive_forward(encoder_outputs,
                                                                current_layer=i,
                                                                attention_mask=extended_attention_mask,
                                                                head_mask=head_mask)

                pooled_output = self.pooler_activation(self.pooler(encoder_outputs[0][:, 0]))
                logits = output_layers[i](pooled_output)
                if regression:
                    labels = logits.detach()
                    if patient_result is not None:
                        patient_labels = patient_result.detach()
                    if (patient_result is not None) and torch.abs(patient_result - labels) < self.regression_threshold:
                        patient_counter += 1
                    else:
                        patient_counter = 0
                else:
                    labels = logits.detach().argmax(dim=1)
                    if patient_result is not None:
                        patient_labels = patient_result.detach().argmax(dim=1)
                    if (patient_result is not None) and torch.all(labels.eq(patient_labels)):
                        patient_counter += 1
                    else:
                        patient_counter = 0

                patient_result = logits
                if patient_counter == self.patience:
                    break
            res = [patient_result]
            self.inference_layers_num += calculated_layer_num
            self.inference_instances_num += 1

        return res


class AlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, self.config.num_labels) for _ in range(config.num_hidden_layers)])

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        logits = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_dropout=self.dropout,
            output_layers=self.classifiers,
            regression=self.num_labels == 1
        )

        outputs = (logits[-1],)

        if labels is not None:
            total_loss = None
            total_weights = 0
            for ix, logits_item in enumerate(logits):
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits_item.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits_item.view(-1, self.num_labels), labels.view(-1))
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss * (ix + 1)
                total_weights += ix + 1
            outputs = (total_loss / total_weights,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

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
        self.albert.set_patience(-666)
        res = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_dropout=self.dropout,
            output_layers=self.classifiers,
            regression=self.num_labels == 1
        )
        highway_logits = res['highway_logits']
        highway_pooler_outputs = res['highway_pooler_outputs']
        exit_layers, exit_logits, predictions = batch_compute_early_exit_layer_and_prediction(
            highway_logits=highway_logits,
            early_exit_patience=early_exit_patience,
        )
        rtn = {
            'exit_logits': exit_logits, # batch_size * num_classes
            'highway_logits': highway_logits, # batch_size * num_layers * num_classes
            'highway_pooler_outputs': highway_pooler_outputs, # batch_size * num_layers * hidden_dim
            'exit_layers': exit_layers, # batch_size
            'predictions': predictions, # batch_size
        }
        return rtn

