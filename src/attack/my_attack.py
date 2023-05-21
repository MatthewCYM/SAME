# from textattack import Attack
from textattack.attack_recipes import AttackRecipe
import numpy as np
import torch
from typing import List, Union
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    MaxModificationRate,
    RepeatModification,
    StopwordModification,
)
from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)
from textattack.constraints import Constraint, PreTransformationConstraint
import textattack
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.shared import utils
from textattack.transformations import Transformation
from textattack.goal_functions import GoalFunction
from textattack.goal_function_results import ClassificationGoalFunctionResult
from textattack.transformations import (
    CompositeTransformation,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwap
)
import random
import logging
import string
import lru
from .my_constraints import MyMaxWordsPerturbed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def construct_mutants(original_text, idx, word):
    return original_text.replace_words_at_indices([idx], [word])


class MyWordSwap(WordSwap):
    def __init__(self, model_wrapper, top_n=1):
        self.model = model_wrapper.model
        self.model_wrapper = model_wrapper
        self.tokenizer = self.model_wrapper.tokenizer

        self.top_n = top_n
        self.is_black_box = False

    def convert_single_token_to_id(self, attacked_text, ori_word_idx, token):
        if 'albert' in self.model_wrapper.model_type:
            word_idx = self.tokenizer.convert_tokens_to_ids('▁' + token)
        elif 'roberta' in self.model_wrapper.model_type:
            text_idx = attacked_text._text_index_of_word_index(ori_word_idx)
            if text_idx != 0 and attacked_text.text[text_idx - 1] == " ":
                word_idx = self.tokenizer.convert_tokens_to_ids('Ġ' + token)
            else:
                word_idx = self.tokenizer.convert_tokens_to_ids(token)
        else:
            word_idx = self.tokenizer.convert_tokens_to_ids(token)
        return word_idx

    def convert_single_id_to_token(self, word_idx):
        seed_word = self.tokenizer.convert_ids_to_tokens(word_idx)
        if 'albert' in self.model_wrapper.model_type:
            seed_word = seed_word.replace('▁', '')
        elif 'roberta' in self.model_wrapper.model_type:
            seed_word = seed_word.replace('Ġ', '')
        else:
            pass
        return seed_word


class MyTokenMutationGradientBased(MyWordSwap):
    def __init__(self, model_wrapper, top_n=1):
        super(MyTokenMutationGradientBased, self).__init__(model_wrapper, top_n)

    def _get_replacement_words_by_grad(self, attacked_text, indices_to_replace):
        lookup_table = self.model.get_input_embeddings().weight.data.cpu()

        grad_output = self.model_wrapper.get_grad(attacked_text.tokenizer_input)
        emb_grad = torch.tensor(grad_output["gradient"])

        text_ids = grad_output["ids"]
        vocab_size = lookup_table.size(0)
        diffs = torch.zeros(len(indices_to_replace), vocab_size)
        indices_to_replace = list(indices_to_replace)

        for j, ori_word_idx in enumerate(indices_to_replace):
            word_ = attacked_text.words[ori_word_idx]
            word_idx = self.convert_single_token_to_id(attacked_text, ori_word_idx, word_)
            # Make sure the word is in bounds.
            if word_idx not in text_ids.tolist()[0]:
                continue
            if word_idx >= len(emb_grad):
                continue

            # Get the grad w.r.t the one-hot index of the word.
            b_grads = lookup_table.mv(emb_grad[word_idx]).squeeze()
            a_grad = b_grads[word_idx]
            diffs[j] = b_grads - a_grad

        # Don't change to the special token.
        diffs[:, self.tokenizer.pad_token_id] = float("inf")
        diffs[:, self.tokenizer.cls_token_id] = float("inf")
        diffs[:, self.tokenizer.sep_token_id] = float("inf")
        diffs[:, self.tokenizer.mask_token_id] = float("inf")

        # Find best indices within 2-d tensor by flattening.
        word_idxs_sorted_by_grad = (diffs).flatten().argsort()

        candidates = []
        num_words_in_text, num_words_in_vocab = diffs.shape
        for idx in word_idxs_sorted_by_grad.tolist():
            idx_in_diffs = idx // num_words_in_vocab
            idx_in_vocab = idx % (num_words_in_vocab)
            idx_in_sentence = indices_to_replace[idx_in_diffs]
            word = self.tokenizer.convert_ids_to_tokens(idx_in_vocab)

            if word.startswith('##'):
                continue
            if 'albert' in self.model_wrapper.model_type:
                if not word.startswith('▁'):
                    continue
                word = word.strip('▁')
                if len(word) == 0:
                    continue
            if 'roberta' in self.model_wrapper.model_type:
                text_idx_in_sentence = attacked_text._text_index_of_word_index(idx_in_sentence)

                if text_idx_in_sentence != 0 and attacked_text.text[text_idx_in_sentence - 1] == " ":
                    # if is space
                    if not word.startswith('Ġ'):
                        continue
                    word = word.strip('Ġ')
                else:
                    if word.startswith('Ġ'):
                        continue
                if len(word) == 0:
                    continue

            if (not utils.has_letter(word)) or (len(utils.words_from_text(word)) != 1):
                # Do not consider words that are solely letters or punctuation.
                continue
            candidates.append((word, idx_in_sentence))
            if len(candidates) == self.top_n:
                break

        return candidates

    def _get_transformations(self, attacked_text, indices_to_replace):

        transformations = []
        for word, idx in self._get_replacement_words_by_grad(
                attacked_text, indices_to_replace
        ):
            new_text = construct_mutants(attacked_text, idx, word)
            transformations.append(new_text)

        return transformations


class MyCharacterMutationGradientBased(MyWordSwap):
    def __init__(self, model_wrapper, top_n, top_k):
        super(MyCharacterMutationGradientBased, self).__init__(model_wrapper, top_n)
        self.letters_to_insert = string.ascii_letters
        self.top_k = top_k

    def _get_transformations(self, attacked_text, indices_to_replace):
        transformations = []
        for word, idx in self._get_replacement_words_by_grad(
                attacked_text, indices_to_replace
        ):
            new_text = construct_mutants(attacked_text, idx, word)
            transformations.append(new_text)
        return transformations

    def _mutate(self, word):
        func_list = [
            self._get_neighbor_swap_words,
            self._get_character_insert_words,
            self._get_character_delete_words,
            self._get_homoglyph_replace_words
        ]
        mutants = []
        for func in func_list:
            candidates = func(word)
            mutants.extend(candidates)
        return mutants

    def _get_neighbor_swap_words(self, word):
        """Returns a list containing all possible words with 1 pair of
        neighboring characters swapped."""

        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 0
        end_idx = (len(word) - 1)

        if start_idx >= end_idx:
            return []

        for i in range(start_idx, end_idx):
            candidate_word = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
            candidate_words.append(candidate_word)

        return candidate_words

    def _get_character_insert_words(self, word):
        """Returns returns a list containing all possible words with 1 random
        character inserted."""
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 0
        end_idx = len(word)

        if start_idx >= end_idx:
            return []

        for i in range(start_idx, end_idx):
            candidate_word = word[:i] + self._get_random_letter() + word[i:]
            candidate_words.append(candidate_word)

        return candidate_words

    def _get_character_delete_words(self, word):
        """Returns returns a list containing all possible words with 1 letter
        deleted."""
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 0
        end_idx = len(word)

        if start_idx >= end_idx:
            return []
        for i in range(start_idx, end_idx):
            candidate_word = word[:i] + word[i + 1 :]
            candidate_words.append(candidate_word)

        return candidate_words

    def _get_homoglyph_replace_words(self, word):
        """Returns a list containing all possible words with 1 character
        replaced by a homoglyph."""
        candidate_words = []
        homos = {
            "-": "˗",
            "9": "৭",
            "8": "Ȣ",
            "7": "𝟕",
            "6": "б",
            "5": "Ƽ",
            "4": "Ꮞ",
            "3": "Ʒ",
            "2": "ᒿ",
            "1": "l",
            "0": "O",
            "'": "`",
            "a": "ɑ",
            "b": "Ь",
            "c": "ϲ",
            "d": "ԁ",
            "e": "е",
            "f": "𝚏",
            "g": "ɡ",
            "h": "հ",
            "i": "і",
            "j": "ϳ",
            "k": "𝒌",
            "l": "ⅼ",
            "m": "ｍ",
            "n": "ո",
            "o": "о",
            "p": "р",
            "q": "ԛ",
            "r": "ⲅ",
            "s": "ѕ",
            "t": "𝚝",
            "u": "ս",
            "v": "ѵ",
            "w": "ԝ",
            "x": "×",
            "y": "у",
            "z": "ᴢ",
        }
        for i in range(len(word)):
            if word[i] in homos:
                repl_letter = homos[word[i]]
                candidate_word = word[:i] + repl_letter + word[i + 1 :]
                candidate_words.append(candidate_word)

        return candidate_words

    def _get_replacement_words_by_grad(self, attacked_text, indices_to_replace):
        grad_output = self.model_wrapper.get_grad(attacked_text.tokenizer_input)
        emb_grad = torch.tensor(grad_output["gradient"])

        all_sentence_tk = grad_output["ids"].tolist()[0]

        # find critical token: vocab size
        tk_score = -emb_grad.sum(1)
        tk_score = [float(v) if i in all_sentence_tk else float('inf') for i, v in enumerate(tk_score)]
        tk_score = np.array(tk_score)
        tk_rank = tk_score.argsort()
        candidates = []

        for word_position in indices_to_replace:
            token_id = self.convert_single_token_to_id(attacked_text, word_position, attacked_text.words[word_position])
            if token_id not in all_sentence_tk:
                continue
            if token_id in self.tokenizer.all_special_ids:
                continue
            # only keep top rank token id
            if token_id not in tk_rank[:min(self.top_k, len(all_sentence_tk))]:
                continue
            seed_word = attacked_text.words[word_position]
            mutants = self._mutate(seed_word)
            # filter same pair
            mutants = [(d, word_position) for d in mutants if d != seed_word]
            candidates.extend(mutants)

        sample_num = min(len(candidates), self.top_n)

        candidates = random.sample(candidates, sample_num)
        return candidates

    def _get_random_letter(self):
        """Helper function that returns a random single letter from the English
        alphabet that could be lowercase or uppercase."""
        return random.choice(self.letters_to_insert)


class EfficiencyDegradationGoal(GoalFunction):
    def __init__(self, model, model_batch_size=128):
        super(EfficiencyDegradationGoal, self).__init__(model, model_batch_size=model_batch_size)

    def _is_goal_complete(self, model_output, _):
        return model_output['exit_layers'] >= self.model.num_exits

    def _get_score(self, model_output, _):
        return model_output['exit_layers'] / self.model.num_exits

    def get_results(self, attacked_text_list, check_skip=False):
        results = []
        if self.query_budget < float("inf"):
            queries_left = self.query_budget - self.num_queries
            attacked_text_list = attacked_text_list[:queries_left]
        self.num_queries += len(attacked_text_list)

        with torch.no_grad():
            model_outputs = self._call_model(attacked_text_list)

        for attacked_text, raw_output in zip(attacked_text_list, model_outputs):
            displayed_output = self._get_displayed_output(raw_output)
            goal_status = self._get_goal_status(
                raw_output, attacked_text, check_skip=check_skip
            )
            goal_function_score = self._get_score(raw_output, attacked_text)
            results.append(
                self._goal_function_result_type()(
                    attacked_text,
                    raw_output,
                    displayed_output,
                    goal_status,
                    goal_function_score,
                    self.num_queries,
                    self.ground_truth_output,
                )
            )
        return results, self.num_queries == self.query_budget

    def _call_model_uncached(self, attacked_text_list):
        """Queries model and returns outputs for a list of AttackedText
        objects."""
        if not len(attacked_text_list):
            return []

        # inputs = [at.text for at in attacked_text_list]
        inputs = [at.tokenizer_input for at in attacked_text_list]

        outputs = []
        i = 0
        while i < len(inputs):
            batch = inputs[i: i + self.batch_size]
            batch_preds = self.model.prediction(batch)

            # Some seq-to-seq models will return a single string as a prediction
            # for a single-string list. Wrap these in a list.
            batch_preds = [
                {k: batch_preds[k][i:i+1].detach() for k in batch_preds}
                for i in range(len(batch_preds['scores']))
            ]
            outputs.extend(batch_preds)
            i += self.batch_size

        return self._process_model_outputs(attacked_text_list, outputs)

    def _goal_function_result_type(self):
        return ClassificationGoalFunctionResult

    def _process_model_outputs(self, inputs, outputs):
        return outputs


class WhiteBoxSearchAttack:
    def __init__(self,
                 model_wrapper,
                 goal_function: GoalFunction,
                 constraints: List[Union[Constraint, PreTransformationConstraint]],
                 transformation: Transformation,
                 search_config):
        self.model_wrapper = model_wrapper
        self.tokenizer = self.model_wrapper.tokenizer

        self.per_size = search_config['per_size']
        self.beam_width = search_config['beam_width']

        self.goal_function = goal_function
        self.transformation = transformation
        self.is_black_box = False
        self.constraints = []
        self.pre_transformation_constraints = []
        for constraint in constraints:
            if isinstance(
                constraint,
                textattack.constraints.PreTransformationConstraint,
            ):
                self.pre_transformation_constraints.append(constraint)
            else:
                self.constraints.append(constraint)

        if not self.transformation.deterministic:
            self.use_transformation_cache = False
        elif isinstance(self.transformation, CompositeTransformation):
            self.use_transformation_cache = True
            for t in self.transformation.transformations:
                if not t.deterministic:
                    self.use_transformation_cache = False
                    break
        else:
            self.use_transformation_cache = True
        transformation_cache_size = 2 ** 15
        constraint_cache_size = 2 ** 15
        self.transformation_cache_size = transformation_cache_size
        self.transformation_cache = lru.LRU(transformation_cache_size)
        self.constraint_cache_size = constraint_cache_size
        self.constraints_cache = lru.LRU(constraint_cache_size)

    def attack(self, example, ground_truth_output):
        initial_result, _ = self.goal_function.init_attack_example(
            example, ground_truth_output
        )
        if initial_result.goal_status == GoalFunctionResultStatus.SKIPPED:
            result = SkippedAttackResult(initial_result)
            return result

        attack_result = self._attack(initial_result)

        self.clear_cache()
        if attack_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            result = SuccessfulAttackResult(
                initial_result,
                attack_result,
            )
        elif attack_result.goal_status == GoalFunctionResultStatus.SEARCHING:
            result = FailedAttackResult(
                initial_result,
                attack_result,
            )
        elif attack_result.goal_status == GoalFunctionResultStatus.MAXIMIZING:
            result = MaximizedAttackResult(
                initial_result,
                attack_result,
            )
        else:
            raise ValueError(f"Unrecognized goal status {attack_result.goal_status}")

        return result

    def _attack(self, initial_result):

        beam = [initial_result.attacked_text]
        best_result = initial_result

        for per in range(self.per_size):
            potential_next_beam = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
                potential_next_beam += transformations
            # filter transformation
            potential_next_beam_filtered = []
            for item in potential_next_beam:
                if not isinstance(item.tokenizer_input, tuple):
                    logger.info(f'drop wrong transformation:{item}')
                else:
                    potential_next_beam_filtered.append(item)
            potential_next_beam = potential_next_beam_filtered

            if len(potential_next_beam) == 0:
                return best_result

            results, search_over = self.goal_function.get_results(potential_next_beam)
            scores = np.array([r.score for r in results])
            best_result = results[scores.argmax()]
            if search_over:
                return best_result
            best_indices = (-scores).argsort()[: self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]
        return best_result

    def clear_cache(self, recursive=True):
        self.constraints_cache.clear()
        if self.use_transformation_cache:
            self.transformation_cache.clear()
        if recursive:
            self.goal_function.clear_cache()
            for constraint in self.constraints:
                if hasattr(constraint, "clear_cache"):
                    constraint.clear_cache()

    def _get_transformations_uncached(self, current_text, original_text=None, **kwargs):
        """Applies ``self.transformation`` to ``text``, then filters the list
        of possible transformations through the applicable constraints.

        Args:
            current_text: The current ``AttackedText`` on which to perform the transformations.
            original_text: The original ``AttackedText`` from which the attack started.
        Returns:
            A filtered list of transformations where each transformation matches the constraints
        """
        transformed_texts = self.transformation(
            current_text,
            pre_transformation_constraints=self.pre_transformation_constraints,
            **kwargs,
        )

        return transformed_texts

    def get_transformations(self, current_text, original_text=None, **kwargs):
        """Applies ``self.transformation`` to ``text``, then filters the list
        of possible transformations through the applicable constraints.

        Args:
            current_text: The current ``AttackedText`` on which to perform the transformations.
            original_text: The original ``AttackedText`` from which the attack started.
        Returns:
            A filtered list of transformations where each transformation matches the constraints
        """
        if not self.transformation:
            raise RuntimeError(
                "Cannot call `get_transformations` without a transformation."
            )

        if self.use_transformation_cache:
            cache_key = tuple([current_text] + sorted(kwargs.items()))
            if utils.hashable(cache_key) and cache_key in self.transformation_cache:
                # promote transformed_text to the top of the LRU cache
                self.transformation_cache[cache_key] = self.transformation_cache[
                    cache_key
                ]
                transformed_texts = list(self.transformation_cache[cache_key])
            else:
                transformed_texts = self._get_transformations_uncached(
                    current_text, original_text, **kwargs
                )
                if utils.hashable(cache_key):
                    self.transformation_cache[cache_key] = tuple(transformed_texts)
        else:
            transformed_texts = self._get_transformations_uncached(
                current_text, original_text, **kwargs
            )

        return self.filter_transformations(
            transformed_texts, current_text, original_text
        )

    def _filter_transformations_uncached(
        self, transformed_texts, current_text, original_text=None
    ):
        """Filters a list of potential transformed texts based on
        ``self.constraints``

        Args:
            transformed_texts: A list of candidate transformed ``AttackedText`` to filter.
            current_text: The current ``AttackedText`` on which the transformation was applied.
            original_text: The original ``AttackedText`` from which the attack started.
        """
        filtered_texts = transformed_texts[:]
        for C in self.constraints:
            if len(filtered_texts) == 0:
                break
            if C.compare_against_original:
                if not original_text:
                    raise ValueError(
                        f"Missing `original_text` argument when constraint {type(C)} is set to compare against `original_text`"
                    )

                filtered_texts = C.call_many(filtered_texts, original_text)
            else:
                filtered_texts = C.call_many(filtered_texts, current_text)
        # Default to false for all original transformations.
        for original_transformed_text in transformed_texts:
            self.constraints_cache[(current_text, original_transformed_text)] = False
        # Set unfiltered transformations to True in the cache.
        for filtered_text in filtered_texts:
            self.constraints_cache[(current_text, filtered_text)] = True
        return filtered_texts

    def filter_transformations(
        self, transformed_texts, current_text, original_text=None
    ):
        """Filters a list of potential transformed texts based on
        ``self.constraints`` Utilizes an LRU cache to attempt to avoid
        recomputing common transformations.

        Args:
            transformed_texts: A list of candidate transformed ``AttackedText`` to filter.
            current_text: The current ``AttackedText`` on which the transformation was applied.
            original_text: The original ``AttackedText`` from which the attack started.
        """
        # Remove any occurences of current_text in transformed_texts
        transformed_texts = [
            t for t in transformed_texts if t.text != current_text.text
        ]
        # Populate cache with transformed_texts
        uncached_texts = []
        filtered_texts = []
        for transformed_text in transformed_texts:
            if (current_text, transformed_text) not in self.constraints_cache:
                uncached_texts.append(transformed_text)
            else:
                # promote transformed_text to the top of the LRU cache
                self.constraints_cache[
                    (current_text, transformed_text)
                ] = self.constraints_cache[(current_text, transformed_text)]
                if self.constraints_cache[(current_text, transformed_text)]:
                    filtered_texts.append(transformed_text)
        filtered_texts += self._filter_transformations_uncached(
            uncached_texts, current_text, original_text=original_text
        )
        # Sort transformations to ensure order is preserved between runs
        filtered_texts.sort(key=lambda t: t.text)
        return filtered_texts


class EfficiencyAttackRecipe(AttackRecipe):
    @property
    def is_black_box(self):
        return False

    @staticmethod
    def build(model_wrapper, top_n=None, beam_width=None):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class WhiteBoxTokenMutationAttack(EfficiencyAttackRecipe):
    @staticmethod
    def build(model_wrapper, top_n=None, beam_width=None, per_size=None, modification_rate=0.1):
        transformation = MyTokenMutationGradientBased(model_wrapper, top_n=100 if top_n is None else top_n)

        constraints = []

        # add constraints
        constraints.append(RepeatModification())
        constraints.append(StopwordModification())
        constraints.append(MyMaxWordsPerturbed(min_words_perturbed=3, max_percent_perturbed=modification_rate))

        logger.info(f'{top_n}, {beam_width}, {per_size}')
        goal_function = EfficiencyDegradationGoal(model_wrapper)
        search_config = {
            'per_size': 5 if per_size is None else per_size,
            'beam_width': 3 if beam_width is None else beam_width
        }

        return WhiteBoxSearchAttack(model_wrapper, goal_function, constraints, transformation, search_config)


class WhiteBoxCharacterMutationAttack(EfficiencyAttackRecipe):
    @staticmethod
    def build(model_wrapper, top_n=None, beam_width=None, per_size=None, modification_rate=0.1):
        transformation = MyCharacterMutationGradientBased(
            model_wrapper,
            top_n=200 if top_n is None else top_n,
            top_k=10
        )
        constraints = []

        # add constraints
        constraints.append(RepeatModification())
        constraints.append(StopwordModification())
        constraints.append(MyMaxWordsPerturbed(min_words_perturbed=3, max_percent_perturbed=modification_rate))

        logger.info(f'{top_n}, {beam_width}, {per_size}')

        goal_function = EfficiencyDegradationGoal(model_wrapper)
        search_config = {
            'per_size': 5 if per_size is None else per_size,
            'beam_width': 3 if beam_width is None else beam_width
        }
        return WhiteBoxSearchAttack(model_wrapper, goal_function, constraints, transformation, search_config)
