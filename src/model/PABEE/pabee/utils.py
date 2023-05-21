import torch
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def batch_compute_early_exit_layer_and_prediction(highway_logits, early_exit_patience):
    exit_layers = []
    predictions = []
    exit_logits = []

    # b * n_layer * n_classes
    for batch_idx, highway_logit in enumerate(highway_logits):
        patient_counter = 0
        patient_label = None
        layer_idx = 0
        for layer_idx in range(len(highway_logit)):

            label = torch.argmax(highway_logit[layer_idx])
            if (patient_label is not None) and label == patient_label:
                patient_counter += 1
            else:
                patient_counter = 0
            patient_label = label
            if patient_counter == early_exit_patience:
                break
        exit_layers.append(layer_idx + 1)
        exit_logits.append(highway_logit[layer_idx])
        predictions.append(patient_label)
    exit_layers = torch.Tensor(exit_layers)
    predictions = torch.Tensor(predictions)
    exit_logits = torch.stack(exit_logits)
    return exit_layers, exit_logits, predictions

