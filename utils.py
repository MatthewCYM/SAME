from transformers import (
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    AlbertTokenizer,
    AlbertConfig
)
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from src import DeeBertForSequenceClassification, DeeRobertaForSequenceClassification
from src import PABEEBertForSequenceClassification, PABEEAlbertforSequenceClassification
from src import WhiteBoxTokenMutationAttack
from src import WhiteBoxCharacterMutationAttack

MODEL_CLASSES = {
    "deebert": (BertConfig, DeeBertForSequenceClassification, BertTokenizer),
    "deeroberta": (RobertaConfig, DeeRobertaForSequenceClassification, RobertaTokenizer),
    "pabeebert": (BertConfig, PABEEBertForSequenceClassification, BertTokenizer),
    'pabeealbert': (AlbertConfig, PABEEAlbertforSequenceClassification, AlbertTokenizer)
}

DATASET_LIST = [
    'CoLA', 'MNLI', 'MRPC', 'QNLI',  'QQP',
    'RTE', 'SST-2',
]


BASELINE_LIST = [
    (WhiteBoxTokenMutationAttack, 'get_slow_grad'),
    (WhiteBoxCharacterMutationAttack, 'get_slow_grad'),
]


def load_dataset(args):
    processor = processors[args.task_name]()
    output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()

    if args.task_name in ["mnli", "mnli-mm"] and 'roberta' in args.model_type:
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]

    label_map = {label: i for i, label in enumerate(label_list)}
    train_examples = processor.get_train_examples(args.data_dir)
    test_examples = processor.get_dev_examples(args.data_dir)
    for data in train_examples:
        data.label = label_map[data.label] if output_mode == "classification" else float(data.label)
    for data in test_examples:
        data.label = label_map[data.label] if output_mode == "classification" else float(data.label)
    return train_examples, test_examples


def load_model_dataset(args):
    args.output_mode = output_modes[args.task_name]

    # MODEL_CLASSES = []
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.model_name_or_path,
        cache_dir='./cache'
    )

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir='./cache'
    )

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        cache_dir='./cache'
    )

    # set early exit entropy later
    train_examples, test_examples = load_dataset(args)

    return config, model, tokenizer, train_examples, test_examples
