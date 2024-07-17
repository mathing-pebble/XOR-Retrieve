import logging
import os
import pickle
import sys
import json

import datasets
import jax
import numpy as np
from flax.training.common_utils import shard
from jax import pmap
from arguments import DataArguments
from arguments import TevatronTrainingArguments as TrainingArguments
from arguments import ModelArguments
from data import EncodeCollator, EncodeDataset
from _datasets.dataset import HFQueryDataset, HFCorpusDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from flax.training.train_state import TrainState
from flax import jax_utils
import optax
from transformers import (AutoConfig, AutoTokenizer, FlaxAutoModel,
                          HfArgumentParser, TensorType, FlaxBertModel)

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
     
