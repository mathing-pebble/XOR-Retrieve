import logging
import os
import json
import sys

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
                          HfArgumentParser, TensorType)

from google.cloud import storage

logger = logging.getLogger(__name__)

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    model_path = (
        model_args.model_name_or_path
        if not model_args.untie_encoder
        else f'{model_args.model_name_or_path}/{"query_encoder" if data_args.encode_is_qry else "passage_encoder"}'
    )

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = FlaxAutoModel.from_pretrained(model_path, config=config, from_pt=False)

    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    dataset_class = HFQueryDataset if data_args.encode_is_qry else HFCorpusDataset

    # Set a default cache directory if none is provided
    dataset_cache_dir = data_args.data_cache_dir or model_args.cache_dir
    if not dataset_cache_dir:
        dataset_cache_dir = os.path.join("/tmp/dataset_cache", data_args.dataset_name.replace("/", "_"))
    else:
        dataset_cache_dir = os.path.join(dataset_cache_dir, data_args.dataset_name.replace("/", "_"))

    if not os.path.exists(dataset_cache_dir):
        os.makedirs(dataset_cache_dir)
        encode_dataset = dataset_class(tokenizer=tokenizer, data_args=data_args, cache_dir=dataset_cache_dir)
        encode_dataset = EncodeDataset(encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
                                       tokenizer, max_len=text_max_length)
    else:
        logger.info("Dataset already downloaded. Reusing the cache.")
        encode_dataset = dataset_class(tokenizer=tokenizer, data_args=data_args, cache_dir=dataset_cache_dir)
        encode_dataset = EncodeDataset(encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
                                       tokenizer, max_len=text_max_length)

    dataset_size = len(encode_dataset)
    padding_prefix = "padding_"
    total_batch_size = len(jax.devices()) * training_args.per_device_eval_batch_size
    features = list(encode_dataset.encode_data.features.keys())
    padding_batch = {features[0]: [], features[1]: []}
    for i in range(total_batch_size - (dataset_size % total_batch_size)):
        padding_batch["text_id"].append(f"{padding_prefix}{i}")
        padding_batch["text"].append([0])
    padding_batch = datasets.Dataset.from_dict(padding_batch)
    encode_dataset.encode_data = datasets.concatenate_datasets([encode_dataset.encode_data, padding_batch])

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size * len(jax.devices()),
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max
