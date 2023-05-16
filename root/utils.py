import os
import random
import numpy as np
import pandas as pd
import logging

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


# def read_examples(filename):
#     """Read examples from filename."""
#     examples = []
#     df = pd.read_csv(filename)

#     code = df['raw_code'].tolist()
#     nl = df['raw_nl'].tolist()
#     for i in range(len(code)):
#         examples.append(
#             Example(
#                 idx=i,
#                 source="[RAW] "+nl[i],
#                 target=code[i],
#             )
#         )
    
#     return examples


# def read_examples_train(filename):
#     """Read examples from filename."""
#     examples = []
#     df = pd.read_csv(filename)
    
#     code = df['temp_code'].tolist()
#     nl = df['temp_nl'].tolist()
#     for i in range(len(code)):
#         examples.append(
#             Example(
#                 idx=i,
#                 source="[TEMPLATE] "+ nl[i],
#                 target=code[i],
#             )
#         )
    
#     code = df['raw_code'].tolist()
#     nl = df['raw_nl'].tolist()
#     for i in range(len(code)):
#         examples.append(
#             Example(
#                 idx=i,
#                 source="[RAW] "+nl[i],
#                 target=code[i],
#             )
#         )
#     return examples

def read_examples(filename):
    """Read examples from filename."""
    examples = []
    df = pd.read_csv(filename)

    code = df['raw_code'].tolist()
    nl = df['raw_nl'].tolist()
    for i in range(len(code)):
        examples.append(
            Example(
                idx=i,
                source= nl[i],
                target=code[i],
            )
        )
    
    return examples


def read_examples_train(filename):
    """Read examples from filename."""
    examples = []
    df = pd.read_csv(filename)
    
    code = df['raw_code'].tolist()
    nl = df['raw_nl'].tolist()
    for i in range(len(code)):
        examples.append(
            Example(
                idx=i,
                source= nl[i],
                target=code[i],
            )
        )
    return examples



class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids


def convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length, stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples, desc='convert examples to features...')):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:max_source_length - 5]
        source_tokens = [tokenizer.cls_token, "<encoder-decoder>", tokenizer.sep_token] + source_tokens + ["<mask0>",tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length - 2]
        target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length

        if example_index < 3:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids
            )
        )
    return features


def set_seed(seed=29):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True