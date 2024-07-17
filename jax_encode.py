import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from flax.training import train_state
from flax.training.common_utils import shard
from transformers import FlaxAutoModel, AutoTokenizer
from tqdm import tqdm
import torch

def unshard(x):
    return np.array(x).reshape((-1,) + x.shape[2:])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--encoded_save_path", type=str, required=True)
    parser.add_argument("--encode_is_qry", action='store_true')
    parser.add_argument("--dataset_proc_num", type=int, default=1)
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset_name)
    
    if args.encode_is_qry:
        dataset = dataset['train']  # Adjust according to the split you want to use
    else:
        dataset = dataset['train']  # Use 'train' split as 'validation' might not exist

    # Load model and tokenizer
    model_path = os.path.abspath(args.model_name_or_path)
    model = FlaxAutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(model_path))

    # Preprocess function
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    
    dataset = dataset.map(preprocess_function, batched=True, num_proc=args.dataset_proc_num)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Create data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.per_device_eval_batch_size)

    def encode(batch):
        input_ids = shard(jax.numpy.array(batch['input_ids']))
        attention_mask = shard(jax.numpy.array(batch['attention_mask']))
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return unshard(outputs[0])  # Assuming outputs[0] is the tensor of interest

    # Encode all examples
    all_embeddings = []
    for batch in tqdm(dataloader):
        embeddings = encode(batch)
        all_embeddings.append(embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Save embeddings
    with open(args.encoded_save_path, 'wb') as f:
        pickle.dump((all_embeddings, list(range(len(all_embeddings)))), f)

if __name__ == "__main__":
    main()
