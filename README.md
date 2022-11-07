# DA4JIE

This repository contains the official implementation of:

**Unsupervised Domain Adaptation for Joint Information Extraction**  
*Nghia Trung Ngo, Bonan Min, Thien Huu Nguyen*  
ACL 2022 | [Paper](https://arxiv.org/abs/xxxx.xxxxx)  
University of Oregon - Amazon AWS AI Labs

---

## Abstract

Joint Information Extraction (JIE) aims to jointly solve multiple tasks in the Information Extraction pipeline (entity mention, event trigger, relation, and event argument extraction). Due to their ability to leverage task dependencies and avoid error propagation, JIE models have presented state-of-the-art performance for different IE tasks. However, current JIE methods only focus on the standard supervised learning setting where training and test data come from the same domain. Cross-domain learning has not been explored for JIE, thus hindering its practical application. To address this issue, our work introduces the first study to evaluate JIE models in the unsupervised domain adaptation (UDA) setting. We present DA4JIE, a novel method to induce domain-invariant representations for JIE. DA4JIE proposes an **Instance-relational Domain Adaptation (IrDA)** mechanism that aligns task-instance representations across domains through a generalized domain-adversarial learning approach. We further devise a **Context-invariant Structure Learning (CiSL)** technique to filter domain-specialized contextual information from induced representations. Extensive experiments demonstrate that DA4JIE significantly improves out-of-domain performance for current state-of-the-art JIE systems across all IE tasks.

---

## Problem Setup

JIE composes of four tasks:

| Task | Abbr. | Description |
|---|---|---|
| Entity Mention Extraction | EME | Detect and classify entity mentions (names, nominals, pronouns) into predefined semantic classes (e.g., *Person*, *Organization*) |
| Event Trigger Detection | ETD | Identify and classify event-triggering words/phrases that evoke a predefined event class (e.g., *Attack*, *Die*) |
| Relation Extraction | RE | Predict semantic relationships between pairs of entity mentions |
| Event Argument Extraction | EAE | Given an event trigger, predict the role each entity mention plays in the corresponding event |

In the **UDA setting**, training uses a labeled source dataset S (N_s samples) and an unlabeled target set T (N_t samples). The goal is to leverage both to maximize performance on test data from the target domain. At each training iteration, a mini-batch contains samples from both S and T: source samples optimize the main downstream tasks, while target samples impose a domain-invariant constraint on the extracted features.

---

## Method

### Overall Architecture

Given an input sentence, the model identifies spans (entity mentions or event triggers), computes their representations, and classifies them for downstream tasks:

1. A **Transformer encoder** (BERT-large-cased) maps each input sentence to contextual word representations X.
2. The **CiSL module** augments X with domain-independent structural features to produce X_ci.
3. Two **CRF layers** (one for event triggers, one for entity mentions) take X_ci as input and output BIO tag sequences identifying spans.
4. Instance representations E_ar (entity) and E_tr (trigger) are computed by CiG-conditioned pooling over detected spans.
5. Separate **task-specific feed-forward classifiers** compute label scores from E_ar, E_tr, (E_ar, E_ar) pairs for RE, and (E_tr, E_ar) pairs for EAE.
6. The **IrDA module** takes representations from both domains and performs adversarial alignment.

The architecture is applied to mini-batches drawn from both source S and target T at training time.

### Instance-relational Domain Adaptation (IrDA)

Standard DANN enforces uniform alignment across all domains, ignoring topological structure among instance types. IrDA generalizes DANN by introducing a **type-relational graph** G_r = (V_r, A_r) over four instance types: entity mentions and event triggers from source and target domains.

- Vertex set: V_r = {E^s_ar, E^s_tr, E^t_ar, E^t_tr}
- Adjacency A_r in R^{4x4}: encodes which type pairs should be aligned (1 = align)
- DA4JIE uses a **chain** connection: E^s_ar -- E^s_tr -- E^t_tr -- E^t_ar

The minimax objective:

```
min_{E,F} max_{D}  L^s_c(E, F) - lambda * L^{s-t}_d(D_g, E)
```

The graph discriminator D_g computes pairwise relationships a_hat_ij = e_i^T e_j between instance type representations, then minimizes:

```
L^{s-t}_d = sum_{i,j} [ -a_ij * log(sigma(a_hat_ij)) - (1 - a_ij) * log(1 - sigma(a_hat_ij)) ]
```

where a_ij is the edge value from A_r. D_g learns to recover G_r; the encoder E learns to prevent it. At equilibrium, connected types are aligned -- their representations contain no domain-identifying information.

**Why chain?**
- Event triggers are tightly tied to predefined event classes shared across domains -- they should be directly aligned across domains.
- Event arguments (entity mentions) are more diverse and context-dependent -- not directly aligned, but *implicitly* aligned via the trigger nodes ("weak" alignment following Xu et al., 2022).
- The trigger-argument edges also implicitly align trigger-argument relational representations, aiding role classification transfer.

### Context-invariant Structure Learning (CiSL)

DANN's target performance bound (Ben-David et al., 2010) includes the joint error of an ideal model performing well on *both* domains. In JIE, this term is non-negligible -- if unconstrained, adversarial training has little alignment effect and worsens the joint error. CiSL produces more transferable representations to minimize this component.

**Step 1: Attention-augmented dependency graphs**

A dependency graph G_d is built from Trankit: word-level nodes with learnable dependency-relation embeddings; binary adjacency A_d[i,j] = 1 if w_j governs w_i.

An attention graph at BERT layer l uses position embeddings as node features and the attention probability matrix as adjacency A^l_a.

Augmented graph at layer l:
```
A^l_da = alpha^l_a * A^l_a  +  alpha^l_d * A_d
Z^l_da = beta^l_a  * Z^l_a  +  beta^l_d  * Z_d
```
where {alpha, beta} are learnable weights. Node features contain no word embeddings -- domain-independent by construction.

**Step 2: GTN fusion**

A Graph Transformer Network (GTN; Yun et al., 2019) fuses {G^l_da}_{l=1..L} across all BERT layers into a single context-invariant graph G_ci = (V_ci, A_ci), with A_ci in R^{nxn} and node features Z_ci in R^{nxh}.

**Step 3: Inject into encoder**

```
X_ci = X + Z_ci
```
Fed to CRF span detectors, encouraging them to use domain-invariant structural features rather than domain-specific surface context.

**Step 4: CiG-conditioned pooling**

Prior work computes instance representations as E = S_base^T * X, where S_base is the binary CRF assignment matrix. DA4JIE learns a conditioned assignment:

```
S_ci = gamma * S_base + mu,    (gamma, mu) = GCN(Z_ci, A_ci)
```

Final representation: E = (gamma * S_base)^T X_ci + mu^T X_ci.

This aggregates information over all sentence words (via mu) while suppressing domain-specific span words (via gamma).

---

## Baselines

| System | Description |
|---|---|
| **BERT** | Shared Transformer encoder for all four IE tasks; task-specific classifiers trained on label distributions |
| **OneIE** (Lin et al., 2020) | BERT + predefined global features capturing cross-subtask and cross-instance interactions |
| **FourIE** (Nguyen et al., 2021) | BERT + graph structure over contextual representations to capture task interactions, with dependency-based regularization |

OneIE and FourIE are the current state-of-the-art JIE systems in the standard supervised setting.

---

## Installation

Requires Python 3.7+, PyTorch >= 1.8, CUDA 10.2+. All experiments run on a single Tesla V100-SXM2 (32 GB), approximately 30 min/epoch per target domain.

```bash
pip install -r requirements.txt
pip install -e .
```

> The code was developed against `transformers==3.5.1`. Pin to this version to avoid model-loading incompatibilities.

---

## Data Preparation

All scripts produce OneIE-format JSON (`train.oneie.json`, `dev.oneie.json`, `test.oneie.json`).

### ACE-05 (LDC2006T06)

ACE-05 contains 599 documents with annotations for 33 event classes, 7 entity classes, 6 relation classes, and 22 argument roles, from 6 domains: `bn`, `nw`, `bc`, `cts`, `wl`, `un`.

**UDA split used in experiments:**
- Source (`in`): `bn` + `nw` -- 80% train, 20% dev.
- Targets: `bc`, `cts`, `wl`, `un` each independently -- 20% unlabeled for training, 80% as test set.

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

### DyGIE++ format

```bash
python preprocessing/process_dygiepp.py \
    -i /path/to/input.json -o data/ace05-e+/train.oneie.json
```

See [data/README.md](data/README.md) for the full expected directory layout.

---

## Training

```bash
python train.py -c configs/example_agie.json    # DA4JIE / AGIE (full model)
python train.py -c configs/example_fourie.json  # FourIE baseline
python train.py -c configs/example_oneie.json   # OneIE baseline
```

### Key Config Fields

| Field | Description |
|---|---|
| `train_file` | Path to `train.oneie.json` |
| `dev_file` | Path to `dev.oneie.json` |
| `test_file` | Path to `test.oneie.json` |
| `log_path` | Output directory for checkpoints and logs |
| `valid_pattern_path` | Path under `resource/valid_patterns/` |
| `bert_cache_dir` | Local BERT model cache |

AGIE-specific fields (`grda.*`, `gtn.*`) are documented in `configs/example_agie.json`. Checkpoints are written to `<log_path>/<timestamp>/`.

### Hyperparameters (Best Configuration)

| Hyperparameter | Best Value | Search Range |
|---|---|---|
| Learning rate (Adam) | 1e-5 | [5e-5, 1e-4, 5e-4, 1e-3, 5e-3] |
| Batch size (50% target data) | 16 | [16, 32, 64] |
| GCN layers | 3 | [2, 3] |
| GTN channels | 4 | [2, 4, 8] |
| FF network hidden dims | [200, 100, 50] | [100, 50] or [200, 100, 50] |
| IrDA balancing lambda | 1 | [0.1, 0.5, 1, 5, 10] |
| Training epochs | 50 | -- |
| Model selection criterion | Best avg. F1 on in-domain dev | -- |

All reported results are averages of **3 runs** with different random seeds.

---

## Inference

```bash
python predict.py \
    -m outputs/<timestamp>/best.role.mdl \
    -i data/input/ \
    -o data/output/ \
    --format ltf --gpu
```

| `--format` | Extension | Description |
|---|---|---|
| `txt` | `.txt` | Plain text, one sentence per line |
| `ltf` | `.ltf.xml` | DARPA LORELEI Translation Format |
| `json` | `.json` | JSON lines |

Output format (one JSON object per line):

```json
{
  "doc_id": "...", "sent_id": "...", "tokens": [...],
  "graph": {
    "entities":  [[start, end, type, mention_type, score], ...],
    "triggers":  [[start, end, type, score], ...],
    "relations": [[ent1_idx, ent2_idx, type, score], ...],
    "roles":     [[trig_idx, ent_idx, role_type, score], ...]
  }
}
```

---

## Results

### Main Results (ACE-05, F1)

In-domain (`in` = `bn`+`nw`) and out-of-domain adaptation to `bc`, `cts`, `wl`, `un`.  
**-I** = identification only (span boundary); **-C** = identification + classification (boundary + type).  
**aTask** = average over 4 classification tasks; **aDom** = average over 4 OOD target domains.

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
| **FourIE** | Trigger-I | 79.1 | 70.7 | 66.0 | 65.2 | 64.3 | 66.6 |
| | Role-I | 66.6 | 60.0 | 52.6 | 48.9 | 49.1 | 52.6 |
| | Entity | 89.1 | 80.3 | 84.4 | 85.4 | 81.9 | 83.0 |
| | Relation-C | 66.0 | 63.7 | 56.6 | 53.1 | 52.7 | 56.5 |
| | Trigger-C | 76.9 | 68.5 | 63.2 | 56.4 | 62.4 | 62.6 |
| | Role-C | 61.8 | 55.4 | 51.8 | 44.5 | 43.6 | 48.8 |
| | **aTask** | 73.5 | 66.9 | 64.0 | 59.8 | 60.1 | 62.7 |
| **DA4JIE** | Trigger-I | 79.0 | 72.2 | 66.0 | 64.4 | 66.5 | **67.3** |
| | Role-I | 67.3 | 59.0 | 54.6 | 49.8 | 51.5 | **53.7** |
| | Entity | 89.2 | 82.6 | 86.0 | 85.2 | 83.0 | **84.2** |
| | Relation-C | 68.8 | 65.3 | 58.7 | 57.7 | 54.3 | **59.0** |
| | Trigger-C | 76.5 | 68.7 | 63.0 | 57.4 | 64.1 | **63.3** |
| | Role-C | 62.5 | 55.6 | 51.9 | 45.3 | 44.4 | **49.3** |
| | **aTask** | **74.2** | **68.0** | **65.0** | **61.4** | **61.4** | **64.0** |

DA4JIE achieves ~2 F1 points above BERT and surpasses current SOTA (FourIE) by over 1 point on average, with simultaneous improvements across all downstream tasks and all OOD domains.

### Ablation Study (avg. task F1, OOD domains)

| Model | `bc` | `cts` | `wl` | `un` | **aDom** |
|---|:-:|:-:|:-:|:-:|:-:|
| **DA4JIE** | **68.0** | **65.0** | **61.4** | **61.4** | **64.0** |
| DA4JIE -CiSL | 67.7 | 61.2 | 59.5 | 60.3 | 62.2 |
| DA4JIE -IrDA | 67.2 | 64.3 | 60.3 | 60.1 | 63.0 |
| DA4JIE -IrDA -CiSL (= BERT) | 66.6 | 63.1 | 59.3 | 59.1 | 62.0 |

CiSL makes instance representations more transferable at low-level, ensuring the necessary condition for IrDA's adversarial training to reach equilibrium. IrDA alone provides consistent domain-level gains. Combined, they especially help on dissimilar target domains (`wl`, `un`).

### Instance-relational Graph Analysis (avg. task F1)

| Graph Topology | `bc` | `cts` | `wl` | `un` | **aDom** |
|---|:-:|:-:|:-:|:-:|:-:|
| **Chain** (DA4JIE) | **68.0** | **65.0** | **61.4** | **61.4** | **64.0** |
| Pair-Task (same task across domains) | 67.5 | 63.2 | 60.0 | 61.1 | 62.9 |
| Full (= standard DANN) | 67.1 | 63.7 | 60.0 | 60.1 | 62.7 |
| Pair-Dom (same domain across tasks) | 67.0 | 62.7 | 59.1 | 59.5 | 62.1 |
| None (no adaptation) | 66.6 | 63.1 | 59.3 | 59.1 | 62.0 |

Full/DANN improves over no adaptation but is overly strict with uniform alignment. Pair-Dom is equivalent to domain conditioning without cross-domain transfer. Chain best reflects the true task-domain topology for JIE.

### CiSL Component Analysis (avg. over OOD domains)

| Model | **aId** | **aCls** |
|---|:-:|:-:|
| **CiSL** (full) | **60.5** | **64.0** |
| CiSL -Pool (no CiG-conditioned pooling) | 59.7 | 63.2 |
| CiSL -Node (no Z_ci to CRF input) | 59.4 | 62.3 |
| CiSL -Node -Pool (disabled) | 58.0 | 62.0 |
| CiSL -Dep (no dependency graph) | 59.8 | 62.8 |
| CiSL -Attn (no attention graphs) | 59.2 | 62.5 |

Node features (Z_ci as CRF input) are the most impactful, boosting both identification and classification by improving low-level representation transferability. Both dependency and attention graphs contribute positively and independently.

---

## Project Structure

```
train.py / predict.py          entry points
configs/                       JSON config files per model variant
da4jie/
  models/  oneie.py  fourie.py  agie.py  model_utils.py
  modules/ bert.py  crf.py  gcn.py  hgcn.py  hgcn_agie.py  gtn.py  graph.py
  config.py  data.py  scorer.py  util.py  vocabs.py
preprocessing/                 ACE / ERE / DyGIE++ converters
resource/                      splits, valid_patterns, type-mapping TSVs
agie/                          original experimental code (legacy reference)
```

---

## Acknowledgements

Supported by ARO grant W911NF-21-1-0112, NSF grant CNS-1747798 (IUCRC Center for Big Learning), and IARPA Contract No. 2019-19051600006 (BETTER Program). Builds on [OneIE](https://blender.cs.illinois.edu/software/oneie/) (Lin et al., 2020) and [FourIE](https://github.com/datquocnguyen/FourIE) (Nguyen et al., 2021). Dependency parsing via [Trankit](https://github.com/nlp-uoregon/trankit). GTN adapted from [Yun et al., 2019](https://arxiv.org/abs/1911.06455).

---

## Citation

```bibtex
@inproceedings{ngo-etal-2022-da4jie,
    title     = {Unsupervised Domain Adaptation for Joint Information Extraction},
    author    = {Ngo, Nghia Trung and Min, Bonan and Nguyen, Thien Huu},
    booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)},
    year      = {2022},
    address   = {Dublin, Ireland},
}
```

## License

MIT. The OneIE codebase this work extends is covered by its own license -- see `agie/`.
