from typing import List


class AbstractAttack:
    def __init__(self):
        pass

    def generate_adv(self, example_x: str):
        pass

    def predict_sentence(self, sents: List[str]):
        model_inputs = self.tokenizer(sents, add_special_tokens=True, return_tensors='pt', padding=True)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        rtn = self.model.batch_adv_forward(**model_inputs, early_exit_entropy=self.early_exit_entropy)

        return rtn
