import logging
import os
import pickle
import sys
import json
import argparse

import datasets
import jax
import numpy as np
from data import EncodeCollator, EncodeDataset
from _datasets.dataset import HFQueryDataset, HFCorpusDataset
from flax.training.common_utils import shard
from jax import pmap
from torch.utils.data import DataLoader
from tqdm import tqdm
from flax.training.train_state import TrainState
from flax import jax_utils
import optax
from transformers import (AutoConfig, AutoTokenizer, FlaxAutoModel,
                          HfArgumentParser, TensorType, FlaxBertModel)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--encoded_save_path", type=str, required=True)
    parser.add_argument("--encode_is_qry", action="store_true")
    parser.add_argument("--dataset_proc_num", type=int, default=1)
    
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    model_path = args.model_name_or_path

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
    )

    model = FlaxBertModel.from_pretrained(model_path, config=config, from_pt=False)

    text_max_length = 128 if args.encode_is_qry else 512
    if args.encode_is_qry:
        encode_dataset = HFQueryDataset(tokenizer=tokenizer, data_args=data_args,
                                        cache_dir=None)
    else:
        encode_dataset = HFCorpusDataset(tokenizer=tokenizer, data_args=data_args,
                                         cache_dir=None)
    encode_dataset = EncodeDataset(encode_dataset.process(1, 0),
                                   tokenizer, max_len=text_max_length)

    # prepare padding batch (for last nonfull batch)
    dataset_size = len(encode_dataset)
    padding_prefix = "padding_"
    total_batch_size = len(jax.devices()) * args.per_device_eval_batch_size
    features = list(encode_dataset.encode_data.features.keys())
    padding_batch = {features[0]: [], features[1]: []}
    for i in range(total_batch_size - (dataset_size % total_batch_size)):
        padding_batch["text_id"].append(f"{padding_prefix}{i}")
        padding_batch["text"].append([0])
    padding_batch = datasets.Dataset.from_dict(padding_batch)
    encode_dataset.encode_data = datasets.concatenate_datasets([encode_dataset.encode_data, padding_batch])

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=args.per_device_eval_batch_size * len(jax.devices()),
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max_length,
            padding='max_length',
            pad_to_multiple_of=16,
            return_tensors=TensorType.NUMPY,
        ),
        shuffle=False,
        drop_last=False,
        num_workers=1,
    )

    adamw = optax.adamw(0.0001)
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)

    def encode_step(batch, state):
        embedding = state.apply_fn(**batch, params=state.params, train=False)[0]
        return embedding[:, 0]

    p_encode_step = pmap(encode_step)
    state = jax_utils.replicate(state)

    encoded = []
    lookup_indices = []

    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        batch_embeddings = p_encode_step(shard(batch.data), state)
        encoded.extend(np.concatenate(batch_embeddings, axis=0))

    output_data = {
        "encoded_queries": [encoded_item.tolist() for encoded_item in encoded[:dataset_size]],
        "lookup_indices": lookup_indices[:dataset_size]
    }

    with open(args.encoded_save_path, 'w') as f:
        json.dump(output_data, f)

if __name__ == "__main__":
    main()
