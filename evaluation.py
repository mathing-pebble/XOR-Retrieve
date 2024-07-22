import pickle
import json
import argparse
from statistics import mean
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
import numpy as np
from faiss_retriever.retriever import FaissRetriever
import tempfile
import os

# Set TMPDIR to /tmp
os.environ['TMPDIR'] = '/tmp'

# Create /tmp if it does not exist
if not os.path.exists('/tmp'):
    os.makedirs('/tmp')

nltk.download('punkt')


def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return np.array(data[0]), data[1]


def read_jsonlines(dataset_name, config_name, split):
    print(f"Loading examples from {dataset_name}, config: {config_name}, split: {split}")
    dataset = load_dataset(dataset_name, config_name, split=split)
    print(f"First item in dataset: {dataset[0]}")
    print(f"Keys in dataset: {dataset[0].keys()}")
    lines = []
    for obj in dataset:
        lines.append(obj)
    return lines


def evaluate_top_k_hit(results, gt_answers, max_token_num=5000):
    per_lang = {}
    for item in tqdm(results):
        q_id = item["id"]
        lang = item["lang"]
        per_lang.setdefault(lang, {"count": 0, "hit": 0})
        ctxs = item["ctxs"]

        if q_id not in gt_answers:
            continue

        answers = gt_answers[q_id]

        span_answers = [answer for answer in answers if answer not in ["yes", "no"]]
        if len(span_answers) == 0:
            continue

        per_lang[lang]["count"] += 1

        concat_string_tokens = []
        for ctx in ctxs:
            tokenized_text = word_tokenize(ctx["text"])
            concat_string_tokens += tokenized_text
            if len(concat_string_tokens) >= max_token_num:
                break
        concat_string_tokens = concat_string_tokens[:max_token_num]
        concat_string = " ".join(concat_string_tokens)
        hit = any(answer in concat_string for answer in span_answers)
        if hit:
            per_lang[lang]["hit"] += 1

    final_results = {lang: result for lang, result in per_lang.items() if result["count"] > 0}

    return final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default=None, type=str, help="Dataset name for HF datasets")
    parser.add_argument("--config", default=None, type=str, help="Dataset configuration name")
    parser.add_argument("--split", default=None, type=str, help="Dataset split")
    parser.add_argument("--query_emb_file", default=None, type=str, help="Path to the query embeddings file (pickle format)")
    parser.add_argument("--corpus_emb_file", default=None, type=str, help="Path to the corpus embeddings file (pickle format)")
    parser.add_argument("--max_token_num", default=5000, type=int, help="Maximum number of tokens to consider")
    parser.add_argument("--top_k", default=5, type=int, help="Top k documents to retrieve")

    args = parser.parse_args()

    query_embeddings, query_lookup = load_embeddings(args.query_emb_file)
    corpus_embeddings, corpus_lookup = load_embeddings(args.corpus_emb_file)

    input_data = read_jsonlines(args.data_file, args.config, args.split)

    # Print structure of the dataset items
    print("Inspecting the dataset structure:")
    for item in input_data[:5]:
        print(item)

    # Create a dictionary of query_id to answers
    qid2answers = {item["query_id"]: item["answers"] for item in input_data}

    # Initialize FAISS retriever
    retriever = FaissRetriever(corpus_embeddings, "Flat")

    # Perform retrieval
    top_k_scores, top_k_indices = retriever.batch_search(query_embeddings, args.top_k, batch_size=128, quiet=False)

    # Prepare results for evaluation
    results = []
    for q_idx, top_k in enumerate(top_k_indices):
        results.append({
            "id": query_lookup[q_idx],
            "lang": input_data[q_idx].get("lang", "unknown"),
            "ctxs": [{"text": corpus_lookup[idx]} for idx in top_k]
        })

    # Evaluate results
    for topk in [2, 5]:
        print(f"Evaluating R@{topk}kt")
        pred_per_lang_results = evaluate_top_k_hit(results, qid2answers, topk * 1000)
        avg_scores = []
        for lang in pred_per_lang_results:
            print(f"Performance on {lang} ({pred_per_lang_results[lang]['count']} examples)")
            per_lang_score = (pred_per_lang_results[lang]["hit"] / pred_per_lang_results[lang]["count"]) * 100
            print(per_lang_score)
            avg_scores.append(per_lang_score)

        print("Final macro averaged score: ")
        print(mean(avg_scores))


if __name__ == "__main__":
    main()
