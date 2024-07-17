import logging
import os
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
        training_args: TrainingArguments

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
        model_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = FlaxAutoModel.from_pretrained(model_path, config=config, from_pt=False)

    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    dataset_class = HFQueryDataset if data_args.encode_is_qry else HFCorpusDataset

    dataset = load_dataset(data_args.dataset_name, cache_dir=model_args.cache_dir)
    dataset = dataset['validation'] if 'validation' in dataset else dataset['train']

    encode_dataset = EncodeDataset(dataset, tokenizer, max_len=text_max_length)
    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size * len(jax.devices()),
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max_length,
            padding='max_length',
            pad_to_multiple_of=16,
            return_tensors=TensorType.NUMPY,
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
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
        "encoded_queries": [encoded_item.tolist() for encoded_item in encoded],
        "lookup_indices": lookup_indices
    }

    with open(data_args.encoded_save_path, 'w') as f:
        json.dump(output_data, f)

if __name__ == "__main__":
    main()
