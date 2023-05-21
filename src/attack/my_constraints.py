

import math
import logging
from textattack.constraints import Constraint
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MyMaxWordsPerturbed(Constraint):
    def __init__(
        self, min_words_perturbed=None, max_percent_perturbed=None, compare_against_original=True
    ):
        super().__init__(compare_against_original)
        if not compare_against_original:
            raise ValueError(
                "Cannot apply constraint MaxWordsPerturbed with `compare_against_original=False`"
            )
        self.min_words_perturbed = min_words_perturbed
        self.max_percent_perturbed = max_percent_perturbed

    def _check_constraint(self, transformed_text, reference_text):
        num_words = min(len(transformed_text.words), len(reference_text.words))
        max_words_perturbed = math.ceil(num_words * self.max_percent_perturbed)
        max_words_perturbed = max(self.min_words_perturbed, max_words_perturbed)

        num_words_diff = len(transformed_text.all_words_diff(reference_text))
        return num_words_diff <= max_words_perturbed

    def extra_repr_keys(self):
        metric = []
        if self.max_percent is not None:
            metric.append("max_percent_perturbed")
        if self.min_words_perturbed is not None:
            metric.append("min_words_perturbed")
        return metric + super().extra_repr_keys()