## SAME
Official implementation of ACL 2023 paper "Dynamic Transformers Provide a False Sense of Efficiency"

### Overview
We propose a simple yet effective energy-oriented attacking framework, **SAME**, a **S**lowdown **A**ttack framework on **M**ulti-**E**xit models.


### Requirements
To run our code, please install all the dependency packages by using the following command:
```shell script
pip install -r requirements.txt
```
and also download and prepare the glue data by using the following command:
```shell script
python tools/download_glue.py
```

### Example Usage
We upload two trained multi-exit models to huggingface hub. More models can be trained using the official repo of DeeBERT and PABEE.

To use **SAME** to attack a entropy-based multi-exit model:
```shell script
python main.py \
        --early_exit_entropy 0.19 \
        --model_name_or_path mattymchen/deebert-base-sst2 \
        --model_type deebert \
        --data_dir glue_data/SST-2 \
        --task_name SST-2 \
        --do_lower_case \
        --lam 0.8 \
        --top_n 100 \
        --beam_width 5 \
        --per_size 10 \
        --output_dir results/deebert-base-sst2
```
To use **SAME** to attack a patience-based multi-exit model:
```shell script
python main.py \
        --early_exit_patience 4 \
        --model_name_or_path mattymchen/pabee-bert-base-sst2 \
        --model_type pabeebert \
        --data_dir glue_data/SST-2 \
        --task_name SST-2 \
        --do_lower_case  \
        --lam 0.8 \
        --top_n 100 \
        --beam_width 5 \
        --per_size 10 \
        --output_dir results/pabee-bert-base-sst2
```

### Citation
Please cite our paper if you are inspired by SAME in your work:
```
@inproceedings{chen2023same,
  title={Dynamic Transformers Provide a False Sense of Efficiency},
  author={Chen, Yiming and Chen, Simin and Li, Zexin and Yang, Wei and Liu, Cong and Tan, Robby and Li, Haizhou},
  booktitle={Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2023}
}
```

### Acknowledgement
Code is implemented based on [TextAttack](https://github.com/QData/TextAttack), [DeeBERT](https://github.com/castorini/DeeBERT), and [PABEE](https://github.com/JetRunner/PABEE). We would like thank the authors for making their code public.
