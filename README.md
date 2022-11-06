# DA4JIE

This repository contains the implementation of:

**Unsupervised Domain Adaptation for Joint Information Extraction**  
*Nghia Trung Ngo, Bonan Min, Thien Huu Nguyen*  
Findings of EMNLP 2022 | [Paper](https://aclanthology.org/2022.findings-emnlp.434/)  
University of Oregon · Amazon AWS AI Labs

---

## Abstract

Joint Information Extraction (JIE) aims to jointly solve multiple tasks in the Information Extraction pipeline — entity mention, event trigger, relation, and event argument extraction. While JIE models achieve strong supervised performance by leveraging task dependencies, they have only been studied in the in-domain setting where training and test data come from the same domain. This work introduces the first study on JIE under **unsupervised domain adaptation (UDA)**, proposing DA4JIE to induce domain-invariant representations for JIE via two complementary mechanisms:

- **Instance-relational Domain Adaptation (IrDA)** — aligns task-instance representations across domains through a graph-structured adversarial learning objective.
- **Context-invariant Structure Learning (CiSL)** — filters domain-specialized contextual information from induced representations to improve transferability.

Extensive experiments show that DA4JIE significantly improves out-of-domain performance across all IE tasks.

---

## Problem Setup

JIE composes of four tasks:

| Task | Abbr. | Description |
|---|---|---|
| Entity Mention Extraction | EME | Detect and classify entity mentions (names, nominals, pronouns) |
| Event Trigger Detection | ETD | Identify and classify event-triggering words/phrases |
| Relation Extraction | RE | Predict semantic relationships between pairs of entity mentions |
| Event Argument Extraction | EAE | Given an event trigger, predict the role each entity mention plays |

In the **UDA setting**, training uses a labeled source dataset **S** and an unlabeled target set **T**. At each training iteration, source samples optimize the main downstream tasks while target samples impose a domain-invariant constraint on the extracted features.

---

## Method

### Overall Architecture

Given an input sentence:
1. A **Transformer encoder** (BERT-large-cased) maps each token to contextual representations **X**.
2. The **CiSL module** augments **X** with domain-independent structural features to produce **X**_ci.
3. Two **CRF layers** take **X**_ci as input and identify entity mention and event trigger spans.
4. Instance representations **E**_ar (entity) and **E**_tr (trigger) are computed by CiG-conditioned pooling over detected spans.
5. Task-specific feed-forward classifiers produce label scores for each subtask (NER, RE, trigger detection, argument role labeling).
6. The **IrDA module** takes representations from both domains and performs adversarial alignment.

### Instance-relational Domain Adaptation (IrDA)

Standard DANN enforces uniform alignment across all instance types. IrDA introduces a **type-relational graph** G_r over four node types: entity mentions and event triggers from source and target domains.

DA4JIE uses a **chain topology**:

> E_ar^src — E_tr^src — E_tr^tgt — E_ar^tgt

The graph discriminator D_g learns to recover the adjacency structure; the encoder learns to prevent it. Connected type pairs end up aligned — their representations become domain-invariant.

**Why chain?** Event triggers are tied to shared event classes and benefit from direct cross-domain alignment. Entity mentions are more diverse, so they are implicitly aligned via the trigger nodes (weak alignment). This topology best reflects the true structure of JIE instances.

The training objective:

> min_{E,F} max_{D} L^src_task(E, F) − λ · L^{src→tgt}_disc(D_g, E)

### Context-invariant Structure Learning (CiSL)

CiSL reduces domain-specific surface bias in representations to ensure IrDA's adversarial training can reach equilibrium.

**Step 1: Attention-augmented dependency graphs**

At each BERT layer *l*, an augmented graph combines dependency-parse structure and self-attention probabilities:

> **A**^l_aug = α^l_attn · **A**^l_attn + α^l_dep · **A**_dep  
> **Z**^l_aug = β^l_attn · **Z**^l_attn + β^l_dep · **Z**_dep

Node features contain only positional structure — no word embeddings — making them domain-independent by construction.

**Step 2: GTN fusion**

A Graph Transformer Network (GTN; Yun et al., 2019) fuses the per-layer augmented graphs {G^l_aug}_{l=1..L} into a single context-invariant graph G_ci with node features **Z**_ci.

**Step 3: Inject into encoder**

> **X**_ci = **X** + **Z**_ci

Fed to the CRF span detectors, encouraging them to rely on domain-invariant structural signals.

**Step 4: CiG-conditioned pooling**

Prior work computes span representations as **E** = **S**_base^⊤ **X**. DA4JIE learns a structure-conditioned assignment:

> **S**_ci = γ · **S**_base + μ,   (γ, μ) = GCN(**Z**_ci, **A**_ci)  
> **E** = **S**_ci^⊤ **X**_ci

This aggregates all sentence words (via μ) while suppressing domain-specific span tokens (via γ).

---

## Baselines

| System | Description |
|---|---|
| **BERT** | Shared Transformer encoder with task-specific classifiers |
| **OneIE** (Lin et al., 2020) | BERT + predefined global features capturing cross-subtask interactions |
| **FourIE** (Nguyen et al., 2021) | BERT + graph structure over contextual representations with dependency regularization |

---

## Installation

Requires Python 3.7+, PyTorch ≥ 1.8, CUDA 10.2+.

```bash
pip install -r requirements.txt
pip install -e .
```

> The code was developed against `transformers==3.5.1`. Pin to this version to avoid model-loading incompatibilities.

---

## Data Preparation

All scripts produce OneIE-format JSON (`train.oneie.json`, `dev.oneie.json`, `test.oneie.json`).

### ACE-05 (LDC2006T06)

ACE-05 contains 599 documents from 6 domains: `bn`, `nw`, `bc`, `cts`, `wl`, `un`. UDA split: source = `bn`+`nw`; targets = `bc`, `cts`, `wl`, `un` (each independently).

```bash
python preprocessing/process_ace.py \
    -i /path/to/LDC2006T06/data \
    -o data/ace05-e+ \
    -s resource/splits/ACE05-E+ \
    -b bert-large-cased \
    -c /path/to/bert_cache \
    -l english
```

### ERE (LDC2015E29/E68/E78/E107)

```bash
python preprocessing/process_ere.py \
    -i /path/to/ere/data -o data/ere \
    -b bert-large-cased -c /path/to/bert_cache \
    -l english -d normal
```

---

## Training

```bash
python train.py -c configs/example_agie.json    # DA4JIE / AGIE (full model)
python train.py -c configs/example_fourie.json  # FourIE baseline
python train.py -c configs/example_oneie.json   # OneIE baseline
```

All reported results are averages of **3 runs** with different random seeds. Experiments run on a single Tesla V100-SXM2 (32 GB), approximately 30 min/epoch per target domain.

---

## Inference

```bash
python predict.py \
    -m outputs/<timestamp>/best.role.mdl \
    -i data/input/ \
    -o data/output/ \
    --format ltf --gpu
```

Supported output formats: `txt` (plain text), `ltf` (DARPA LTF XML), `json` (JSON lines).

---

## Results

### Main Results (ACE-05, F1)

In-domain (`in` = `bn`+`nw`) and out-of-domain adaptation to `bc`, `cts`, `wl`, `un`.  
**-I** = identification; **-C** = identification + classification.  
**aTask** = average over 4 classification tasks; **aDom** = average over 4 OOD domains.

| Model | Task | `in` | `bc` | `cts` | `wl` | `un` | **aDom** |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **BERT** | Trigger-I | 78.4 | 71.4 | 65.2 | 62.9 | 66.3 | 66.4 |
| | Role-I | 64.1 | 59.5 | 49.0 | 46.3 | 46.9 | 50.4 |
| | Entity | 88.9 | 80.8 | 84.0 | 85.5 | 80.9 | 82.8 |
| | Relation-C | 64.3 | 61.7 | 58.0 | 52.5 | 48.0 | 55.0 |
| | Trigger-C | 76.3 | 68.7 | 62.4 | 56.3 | 64.5 | 63.0 |
| | Role-C | 60.8 | 55.4 | 47.9 | 42.9 | 43.0 | 47.3 |
| | **aTask** | 72.6 | 66.6 | 63.1 | 59.3 | 59.1 | 62.0 |
| **OneIE** | Trigger-I | 79.1 | 70.3 | 68.2 | 63.2 | 64.6 | 66.6 |
| | Role-I | 66.2 | 60.1 | 51.2 | 50.6 | 46.7 | 52.1 |
| | Entity | 89.1 | 79.5 | 86.9 | 85.5 | 81.5 | 83.4 |
| | Relation-C | 65.6 | 63.1 | 56.7 | 54.7 | 50.0 | 56.1 |
| | Trigger-C | 77.2 | 67.5 | 64.6 | 56.8 | 63.4 | 63.1 |
| | Role-C | 62.2 | 55.7 | 49.9 | 47.2 | 42.6 | 48.9 |
| | **aTask** | 73.5 | 66.5 | 64.6 | 61.1 | 59.4 | 62.9 |
| **FourIE** | **aTask** | 73.5 | 66.9 | 64.0 | 59.8 | 60.1 | 62.7 |
| **DA4JIE** | Trigger-I | 79.0 | 72.2 | 66.0 | 64.4 | 66.5 | **67.3** |
| | Role-I | 67.3 | 59.0 | 54.6 | 49.8 | 51.5 | **53.7** |
| | Entity | 89.2 | 82.6 | 86.0 | 85.2 | 83.0 | **84.2** |
| | Relation-C | 68.8 | 65.3 | 58.7 | 57.7 | 54.3 | **59.0** |
| | Trigger-C | 76.5 | 68.7 | 63.0 | 57.4 | 64.1 | **63.3** |
| | Role-C | 62.5 | 55.6 | 51.9 | 45.3 | 44.4 | **49.3** |
| | **aTask** | **74.2** | **68.0** | **65.0** | **61.4** | **61.4** | **64.0** |

DA4JIE surpasses FourIE by over 1 F1 point on average with consistent improvements across all tasks and OOD domains.

### Ablation Study

| Model | `bc` | `cts` | `wl` | `un` | **aDom** |
|---|:-:|:-:|:-:|:-:|:-:|
| **DA4JIE** | **68.0** | **65.0** | **61.4** | **61.4** | **64.0** |
| DA4JIE −CiSL | 67.7 | 61.2 | 59.5 | 60.3 | 62.2 |
| DA4JIE −IrDA | 67.2 | 64.3 | 60.3 | 60.1 | 63.0 |
| DA4JIE −IrDA −CiSL | 66.6 | 63.1 | 59.3 | 59.1 | 62.0 |

---

## Acknowledgements

Supported by ARO grant W911NF-21-1-0112, NSF grant CNS-1747798, and IARPA Contract No. 2019-19051600006. Builds on [OneIE](https://blender.cs.illinois.edu/software/oneie/) and [FourIE](https://github.com/datquocnguyen/FourIE). Dependency parsing via [Trankit](https://github.com/nlp-uoregon/trankit).

---

## Citation

```bibtex
@inproceedings{ngo-etal-2022-unsupervised,
    title     = {Unsupervised Domain Adaptation for Joint Information Extraction},
    author    = {Ngo, Nghia Trung and Min, Bonan and Nguyen, Thien Huu},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2022},
    year      = {2022},
    address   = {Abu Dhabi, United Arab Emirates},
    publisher = {Association for Computational Linguistics},
    url       = {https://aclanthology.org/2022.findings-emnlp.434},
}
```

## License

MIT
