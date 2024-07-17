# XOR-Retrieve with ContrastiveMix
This repository is the modification of implementation by https://github.com/DoJunggeun/contrastivemix to measure XOR-Retrieve performance of ContrastiveMix retriever, introduced in [ContrastiveMix: Overcoming Code-Mixing Dilemma in Cross-Lingual Transfer for Information Retrieval](https://aclanthology.org/2024.naacl-short.17/). All works are based on concepts and experiments described in [Akari Asai, Jungo Kasai, Jonathan Clark, Kenton Lee, Eunsol Choi, and Hannaneh Hajishirzi. 2021. XOR QA: Cross-lingual Open-Retrieval Question Answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 547–564, Online. Association for Computational Linguistics.]
This work is based on [Tevatron](https://github.com/texttron/tevatron), [Pyserini](https://github.com/castorini/pyserini), dictionaries downloaded from [MUSE](https://github.com/facebookresearch/MUSE) are included in `utils/dict`.



## Requirements
We conducted all experiments on v3-8 TPU VM with Python 3.9.12 and the following dependencies.
```
pip install torch==1.10.1 faiss-cpu==1.7.2 transformers==4.15.0 datasets==1.17.0 pyserini===0.21.0 optax==0.1.5 flax==0.6.11 chex==0.1.8 "jax[tpu]==0.4.7" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Training
### mDPR
```
output_dir=/path/to/output/dir

python jax_train.py \
  --do_train --output_dir ${output_dir} \
  --dataset_name Tevatron/xor-tydi \
  --dataset_language full \
  --model_name_or_path bert-base-multilingual-cased \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 --num_train_epochs 40
```

### NaiveMix
```
lang_abbr=ar  # one of {'ar', 'bn', 'fi', 'id', 'ja', 'ko', 'ru', 'th'}
output_dir=/path/to/output/dir

python jax_train.py \
  --do_train --output_dir ${output_dir} \
  --dataset_name Tevatron/xor-tydi \
  --model_name_or_path bert-base-multilingual-cased \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 --num_train_epochs 40 \
  --codemix_set en-${lang_abbr} \
  --codemix_sentence_ratio 0.2 --codemix_ratio 0.5 
```

### ContrastiveMix
```
lang_abbr=ar
output_dir=/path/to/output/dir

python jax_train.py \
  --do_train --output_dir ${output_dir} \
  --dataset_name Tevatron/xor-tydi \
  --model_name_or_path bert-base-multilingual-cased \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 --num_train_epochs 40 \
  --codemix_set en-${lang_abbr} \
  --codemix_sentence_ratio_query 1 --codemix_ratio_query 0.5 \
  --contrastive --cm_loss_weight 0.1
```

## Evaluation
```
OUTPUT_DIR="Contrastive_output"
CORPUS_DATASET="Tevatron/xor-tydi-corpus"
QUERY_DATASET="Tevatron/xor-tydi:full:dev" # you may chooose xor-tydi:{full, eng-span}:{dev, test}

# Create output directory if it does not exist
mkdir -p ${OUTPUT_DIR}

# encoding corpus
echo "Encoding documents for ${MODEL_DIR}..."
python jax_encode.py \
  --output_dir=temp \
  --model_name_or_path ./${MODEL_DIR}/passage_encoder \
  --per_device_eval_batch_size 156 \
  --dataset_name ${CORPUS_DATASET} \
  --encoded_save_path ${OUTPUT_DIR}/corpus_emb_${MODEL_DIR}.pkl

# encoding query
echo "Encoding queries for ${MODEL_DIR}..."
python jax_encode.py \
  --output_dir=temp \
  --model_name_or_path ./${MODEL_DIR}/query_encoder \
  --per_device_eval_batch_size 1 \
  --dataset_proc_num 4 \
  --dataset_name ${QUERY_DATASET} \
  --encoded_save_path ${OUTPUT_DIR}/query_${MODEL_DIR}.pkl \
  --encode_is_qry


echo "Running evaluation for ${MODEL_DIR}..."
# Run evaluation
python evaluation.py \
  --data_file ${QUERY_DATASET} \
  --pred_file "${OUTPUT_DIR}/query_${MODEL_DIR}.pkl" \
  --max_token_num 5000

echo "Evaluation completed for ${MODEL_DIR}. Results are saved in ${OUTPUT_DIR}/evaluation_${MODEL_DIR}.json"
