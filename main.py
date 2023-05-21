import torch

from utils import load_model_dataset
from argparse import ArgumentParser
import logging
import transformers
import random
import numpy as np
import os
from tqdm import tqdm
import json
import utils
from utils import BASELINE_LIST
from src.model.abstract_model import DynamicTransformerClassifier
import textattack
from textattack.shared import AttackedText
from collections import OrderedDict
from transformers import glue_compute_metrics as compute_metrics
import sys


logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    # for deebert
    parser.add_argument('--early_exit_entropy', type=float, default=None)

    # for pabee
    parser.add_argument('--early_exit_patience', type=int, default=None)

    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='deebert',
                        choices=["deebert", "deeroberta", "pabeebert", "pabeealbert"]
    )
    parser.add_argument('--data_dir', type=str, default='glue_data/SST-2')
    parser.add_argument('--task_name', type=str, default='SST-2')
    parser.add_argument('--do_lower_case', action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument('--demo_size', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='results')

    # hyperparameter for our methods
    parser.add_argument('--top_n', type=int, default=100)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--lam', type=float, default=0.8)
    parser.add_argument('--per_size', type=int, default=10)
    parser.add_argument('--modification_rate', type=float, default=0.1)


    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.task_name = args.task_name.lower()
    args.model_type = args.model_type.lower()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config, model, tokenizer, _, test_data = load_model_dataset(args)
    model = model.eval().to(args.device)

    wrapper_model = DynamicTransformerClassifier(
        model, tokenizer, args.device, args=args
    )

    if args.demo_size is not None:
        test_data = test_data[:args.demo_size]

    logger.info(args)

    for (attack_class, grad_func_name) in BASELINE_LIST:
        ori_exit_layer, adv_exit_layer = 0, 0
        if grad_func_name is not None:
            wrapper_model.get_grad = getattr(wrapper_model, grad_func_name)
            recipe = attack_class.build(wrapper_model,
                                        top_n=args.top_n,
                                        beam_width=args.beam_width,
                                        per_size=args.per_size,
                                        modification_rate=args.modification_rate,
                                        )
        else:
            raise NotImplementedError

        logger.info(attack_class.__name__)
        logger.info(f'is blackbox: {recipe.is_black_box}')
        ori_preds = []
        adv_preds = []
        labels = []
        ori_inputs = []
        adv_inputs = []
        for data in tqdm(test_data):
            x = OrderedDict()
            x['text_a'] = data.text_a
            x['text_b'] = data.text_b if data.text_b else ''
            x = AttackedText(x)
            adv_sample = recipe.attack(x, data.label)

            if grad_func_name is not None:
                with torch.no_grad():
                    ori_rnt = wrapper_model.prediction([adv_sample.original_result.attacked_text.tokenizer_input])
                    adv_rnt = wrapper_model.prediction([adv_sample.perturbed_result.attacked_text.tokenizer_input])
            else:
                with torch.no_grad():
                    ori_rnt = wrapper_model.prediction([x.tokenizer_input])
                    adv_rnt = wrapper_model.prediction([adv_sample])
            ori_exit_layer += int(ori_rnt['exit_layers'][0])
            adv_exit_layer += int(adv_rnt['exit_layers'][0])
            ori_preds.append(int(ori_rnt['predictions'][0]))
            adv_preds.append(int(adv_rnt['predictions'][0]))

            if grad_func_name is not None:
                ori_inputs.append(adv_sample.original_result.attacked_text.tokenizer_input)
                adv_inputs.append(adv_sample.perturbed_result.attacked_text.tokenizer_input)
            else:
                ori_inputs.append(x.tokenizer_input)
                adv_inputs.append(adv_sample)
            labels.append(int(data.label))
        ori_preds, adv_preds, labels = np.array(ori_preds), np.array(adv_preds), np.array(labels)

        logger.info(
            f'{attack_class.__name__} '
            f'exit layer before: {ori_exit_layer / len(test_data)} '
            f'after: {adv_exit_layer / len(test_data)}'
        )

        output_dir = args.output_dir + f'/{attack_class.__name__}'
        if attack_class in [utils.WhiteBoxCharacterMutationAttack, utils.WhiteBoxTokenMutationAttack]:
            output_dir = output_dir + f'-{grad_func_name}'
            output_dir = output_dir + f'-lam={args.lam}-top_n={args.top_n}-beam_width={args.beam_width}-per_size={args.per_size}'

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'ori_inputs.txt'), 'w', encoding='utf-8') as f:
            for line in ori_inputs:
                f.write('\t'.join(line)+'\n')
        with open(os.path.join(output_dir, 'adv_inputs.txt'), 'w', encoding='utf-8') as f:
            for line in adv_inputs:
                f.write('\t'.join(line)+'\n')
        results = {
            'name': f'{attack_class.__name__}',
            'model': f'{args.model_name_or_path}',
            'ori_metrics': compute_metrics(args.task_name, ori_preds, labels),
            'adv_metrics': compute_metrics(args.task_name, adv_preds, labels),
            'ori_speedup': config.num_hidden_layers / (ori_exit_layer / len(test_data)),
            'adv_speedup': config.num_hidden_layers / (adv_exit_layer / len(test_data))
        }
        if 'pabee' not in args.model_type:
            results['entropy'] = f'{args.early_exit_entropy}'
        else:
            results['patience'] = f'{args.early_exit_patience}'
        with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
