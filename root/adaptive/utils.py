import os
import random
import numpy as np
import logging
from random import shuffle, choice, sample
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, mask_ids, label_id):
        self.input_ids = input_ids
        self.mask_ids = mask_ids
        self.label_id = label_id

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, tokenizer):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    vocab_list = list(tokenizer.get_vocab().keys())
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == tokenizer.bos_token or token == tokenizer.eos_token:
            continue
        cand_indices.append(i)

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))

    shuffle(cand_indices)
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            masked_token = tokenizer.mask_token
        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels

def create_examples(data_path, max_seq_length, masked_lm_prob, max_predictions_per_seq, tokenizer):
    """Creates examples for the training and dev sets."""
    examples = []
    max_num_tokens = max_seq_length - 2
    fr = open(data_path, "r")
    for (i, line) in tqdm(enumerate(fr), desc="Creating Example"):
        tokens_a = line.strip("\n").split()[:max_num_tokens]
        tokens = [tokenizer.bos_token] + tokens_a + [tokenizer.eos_token]
        segment_ids = [0 for _ in range(len(tokens_a) + 2)]
        # remove too short sample
        if len(tokens_a) < 5:
            continue
        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, tokenizer)
        example = {
            "tokens": tokens,
            "segment_ids": segment_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels}
        examples.append(example)
    fr.close()
    return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for i, example in tqdm(enumerate(examples), desc="Converting Feature", total=len(examples)):
        tokens = example["tokens"]
        segment_ids = example["segment_ids"]
        masked_lm_positions = example["masked_lm_positions"]
        masked_lm_labels = example["masked_lm_labels"]
        assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_array = np.ones(max_seq_length, dtype=np.int)
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.int)
        mask_array[:len(input_ids)] = 1

        lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-100)
        lm_label_array[masked_lm_positions] = masked_label_ids

        feature = InputFeatures(input_ids=input_array, mask_ids=mask_array,
                                 label_id=lm_label_array)
        features.append(feature)
        if i < 3:
            print("input_ids: %s\nmask_ids: %s\nlabel_id:%s" %(input_array, mask_array, lm_label_array))
    return features


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True