

## Enabling Collaborative Parametric Knowledge Calibration for Retrieval-Augmented Vision Question Answering

This repository contains the code for the UniRVQA framework from the paper (arXiv: [https://arxiv.org/abs/2504.04065], under reivew at Knowledge-based Systems). UniRVQA is a unified multimodal MLLM framework for retrieval-augmented knowledge-based VQA. The code allows the reproduction of experiments on BLIP-2 and InstructBLIP. In the paper, the overall framework is referred to as **UniRVQA**, a unified retrieval-augmented VQA system that couples retrieval and answer generation through collaborative parametric knowledge calibration, late interaction retreival, and reflective answering.


## What this repo does

The training and inference pipeline implemented here is:

1. Modify ColBERT's code (https://github.com/stanford-futuredata/ColBERT) into the UniRVQA's unified training framework.
2. Learn a unified representation with a late-interaction and knowledge-based multimodal generation objectives.
3. Build an index over an external corpus.
4. INFERENCE: adaptive retreival and KB-VQA generation.
5. Use the reflective decision process to determine when parametric knowledge is sufficient and when retrieved evidence should be used.

The repository includes:

- Custom BLIP-2 and InstructBLIP variants in [`model/`](./model)
- A vendored ColBERT implementation in [`model/ColBERT/`](./model/ColBERT)
- Dataset loaders for OK-VQA and InfoSeek in [`model/dataset.py`](./model/dataset.py)
- Training, indexing, retrieval, and prediction entry points at the repo root
- An InfoSeek evaluation toolkit in [`infoseek_eval/`](./infoseek_eval)

## Repository layout

```text
.
├── train.py                 # LoRA fine-tuning
├── index.py                 # Build a ColBERT index over the corpus
├── search.py                # Retrieve passages for validation questions
├── pred.py                  # Generate final answers from retrieval results
├── setup.sh                 # Environment/data bootstrap script
├── job.sh                   # Example commands
├── model/
│   ├── mmRAG_blip2.py
│   ├── mmRAG_instructblip.py
│   ├── dataset.py
│   └── ColBERT/
├── misc/
│   ├── util.py
│   └── eval.py
├── assets/
│   ├── data/
│   ├── checkpoints/
│   └── experiments/
└── infoseek_eval/
```

## Datasets

The code expects data under `assets/data/`:

- `assets/data/okvqa/`
- `assets/data/infoseek/`

For OK-VQA, the loader uses:

- `google_corpus/retriever_train.json`
- `google_corpus/retriever_val.json`
- `google_corpus/okvqa_full_clean_corpus.csv`
- `mscoco_*2014_annotations.json`
- `train2014/` and `val2014/` image folders

For InfoSeek, the loader uses:

- `infoseek_train_subset.jsonl`
- `infoseek_val.jsonl`
- `infoseek_val_2000.jsonl`
- `infoseek_*_caption.json`
- `wiki_100k_short.csv`
- `OVEN_images/<split>/`

`setup.sh` contains one environment bootstrap path for downloading and extracting assets, but it is opinionated and tied to a specific workspace layout. It is best used as a reference and adapted to your machine if needed.

## Environment setup

This project is designed for GPU training/inference with PyTorch, Transformers, PEFT, and FAISS/ColBERT.

Typical setup:

```bash
conda create -n qrag python=3.10 -y
conda activate qrag

pip install torch torchvision torchaudio
pip install transformers datasets peft wandb tqdm pillow pandas
conda install -c pytorch -c nvidia -c conda-forge faiss-gpu blas=*=*mkl gcc=12 gxx=12 -y

pip install -e ./model/ColBERT
```

Notes:

- `train.py` logs to Weights & Biases and currently contains a hardcoded `wandb.login(...)` call. Replace or remove that before running training in your environment.
- The code uses `torch.bfloat16` and assumes CUDA is available for the main workflow.

## Naming convention

Several scripts infer dataset/model names by splitting checkpoint or index names on underscores, so naming matters.

Recommended checkpoint pattern:

```text
<model>_<experiment>_<dataset>
```

Examples:

- `blip2_multi_task_infoseek`
- `instructblip_multi_task_okvqa`

Saved checkpoints then become:

```text
assets/checkpoints/<checkpoint_name>_<step>
```

Examples:

- `assets/checkpoints/instructblip_multi_task_infoseek_3000`
- `assets/checkpoints/blip2_multi_task_okvqa_6000`

## End-to-end workflow

### 1. Train

```bash
python train.py -c instructblip_multi_task_infoseek -b 16 -s 6000 -v 200
```

Important training behavior:

- The base model is selected from the first token of `-c` (`blip2` or `instructblip`)
- The dataset is selected from the last token of `-c` (`okvqa` or `infoseek`)
- Checkpoints are saved to `assets/checkpoints/` every validation interval

Training combines the paper's main learning components:

- Retrieval loss for the late-interaction retriever
- Retrieval-augmented generation loss
- Generation loss without retrieved documents for reflective answering
- Self-verification loss in later training stages

### 2. Build an index

```bash
python index.py -c instructblip_multi_task_infoseek_3000 -d infoseek
```

This builds a ColBERT index over:

- `assets/data/okvqa/google_corpus/okvqa_full_clean_corpus.csv` for OK-VQA
- `assets/data/infoseek/wiki_100k_short.csv` for InfoSeek

The index is written under `assets/experiments/`.

### 3. Search the index

```bash
python search.py -i infoseek_index_instructblip_multi_task_infoseek_3000
```

This runs retrieval on the validation split and writes a processed result CSV under `assets/experiments/search/...`.

It also reports retrieval metrics:

- `PRR@1`, `PRR@5`, `PRR@10`
- `wiki_recall@k` for InfoSeek

### 4. Generate answers

```bash
python pred.py \
  -r assets/experiments/search/infoseek_index_instructblip_multi_task_infoseek_3000/infoseek_index_instructblip_multi_task_infoseek_3000_result_@10.csv \
  -k 5 \
  -c instructblip_multi_task_infoseek_3000
```

This writes predictions as JSONL next to the retrieval results, for example:

```text
assets/experiments/search/.../infoseek_pred_instructblip_multi_task_infoseek_3000.jsonl
```

## Evaluation

Retrieval-side metrics are implemented in [`misc/eval.py`](./misc/eval.py).

For InfoSeek benchmark scoring, see [`infoseek_eval/README.md`](./infoseek_eval/README.md).

## Example commands

More example commands are collected in [`job.sh`](./job.sh).

## Outputs

Common output locations:

- `assets/checkpoints/` for LoRA checkpoints
- `assets/experiments/indexes/` for ColBERT indexes
- `assets/experiments/search/` for retrieval runs and prediction files
- `assets/wandb/` for Weights & Biases logs

## Caveats

- This is research code and assumes a fairly specific local data layout.
- Some scripts rely on filename conventions rather than explicit config files.
- `search.py` currently evaluates on the validation split.
- `pred.py` appends predictions to the output file, so remove old prediction files before rerunning if you want a clean output.

## References

- Paper: *Enabling Collaborative Parametric Knowledge Calibration for Retrieval-Augmented Vision Question Answering* (`arXiv:2504.04065`)
- BLIP-2 / InstructBLIP from Salesforce via Hugging Face Transformers
- ColBERT: included in [`model/ColBERT/`](./model/ColBERT)
- InfoSeek evaluation toolkit in [`infoseek_eval/`](./infoseek_eval)

