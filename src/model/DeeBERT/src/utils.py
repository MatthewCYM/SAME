import torch
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_early_exit_layer(highway_entropy, early_exit_entropy):
    for idx, item in enumerate(highway_entropy):
        if item < early_exit_entropy:
            return idx
    return -1


def batch_compute_early_exit_layer_and_prediction(highway_entropies, early_exit_entropy, model_outputs, highway_logits, num_hidden_layers):
    exit_layers = []
    predictions = []
    exit_logits = []
    for batch_idx, highway_entropy in enumerate(highway_entropies):
        exit_layer = compute_early_exit_layer(highway_entropy, early_exit_entropy)
        if exit_layer == -1:
            # no early exit
            prediction = torch.argmax(model_outputs[0][batch_idx])
            exit_layer = num_hidden_layers - 1
            exit_logit = model_outputs[0][batch_idx]
        else:
            prediction = torch.argmax(highway_logits[batch_idx][exit_layer])
            exit_logit = highway_logits[batch_idx][exit_layer]
        exit_layers.append(exit_layer+1)
        exit_logits.append(exit_logit)
        predictions.append(prediction)
    exit_layers = torch.Tensor(exit_layers)
    predictions = torch.Tensor(predictions)
    exit_logits = torch.stack(exit_logits)
    return exit_layers, exit_logits, predictions
