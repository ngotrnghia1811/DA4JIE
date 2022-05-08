# Data Directory

This directory should contain the preprocessed datasets.

## Expected Structure

```
data/
├── ace05-e+/
│   ├── train.oneie.json
│   ├── dev.oneie.json
│   └── test.oneie.json
├── ace05-r/
│   ├── train.oneie.json
│   ├── dev.oneie.json
│   └── test.oneie.json
├── acm/
│   └── ...
├── dblp/
│   └── ...
├── imdb/
│   └── ...
└── input/         ← raw documents for inference (LTF / TXT)
```

## Obtaining Data

### ACE 2005 (ACE05-E / ACE05-E+ / ACE05-R)

1. Obtain LDC2006T06 from the Linguistic Data Consortium.
2. Run the preprocessing script:

```bash
bash scripts/run_preprocessing_ace.sh
```

See `preprocessing/process_ace.py` for detailed options.

### ERE (LDC2015E29 / LDC2015E68 / LDC2015E78 / LDC2015E107)

```bash
python preprocessing/process_ere.py \
    -i /path/to/ere/data \
    -o data/ere \
    -b bert-large-cased \
    -c /path/to/bert_cache \
    -l english \
    -d normal
```

### DyGIE++ Format

```bash
python preprocessing/process_dygiepp.py \
    -i /path/to/train.json \
    -o data/ace05-e+/train.oneie.json
```

## Domain Adaptation Datasets

The DA4JIE experiments use ACE data as the source domain and can target:
- **ACM** — scientific paper titles
- **DBLP** — bibliography data
- **IMDB** — movie reviews

These non-ACE datasets require annotation conversion. See the paper for details.
