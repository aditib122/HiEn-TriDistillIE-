# HiEn-TriDistillIE

This repository contains a compact implementation and selected experimental results for **HiEn-TriDistillIE for Hindi-English Medical NER and Coarse Relation Extraction**.

## Included configurations

- **2-teacher:** MuRIL + BioBERT → IndicBERTv2 student
- **3-teacher:** MuRIL + BioBERT + PubMedBERT → IndicBERTv2 student
- **4-teacher:** MuRIL + XLM-R / BioBERT + XLM-R → IndicBERTv2 student


## Files

- `HiEn_TriDistillIE_Experiments.ipynb` — notebook containing selected training/evaluation code, reported results, cross-lingual analyses, confusion matrices, and t-SNE visualizations.
- `HiEn_TriDistillIE_Experiments.py` — Python version of the experimental workflow.
- `results/figures/` — selected result figures used for visual inspection.

## Reported analyses

The included results cover selected 2-, 3-, and 4-teacher experiments, repeated-seed robustness, Hindi-only and English-only controls, cross-lingual generalization, human-validated relation extraction, representation diagnostics, confusion matrices, entity-frequency analysis, and t-SNE visualizations.

## Datasets

The English-Ayurveda corpus is derived from the publicly available Charak Samhita resources in the Gita Datasets repository:

https://github.com/gita/Datasets

The Hindi component is obtained from the publicly available Hindi Health Dataset on Kaggle:

https://www.kaggle.com/code/kerneler/starter-hindi-health-dataset-909fee49-0
## Citations
Jain, A., and Arora, A. Named Entity Recognition in Hindi Using Hyperspace Analogue to Language and Conditional Random Field. Pertanika Journal of Science and Technology, UPM, vol. 26, no. 4, pp. 1801-1822, 2018.
Jain, A., Tayal, D.K., and Arora, A. OntoHindi NER- An Ontology Based Novel Approach For Hindi Named Entity Recognition. International Journal of Artificial Intelligence, vol. 16, no. 2, pp. 1-36, 2018.

## Notes on reproducibility

Some heavy training cells were originally developed in Google Colab and may contain Google Drive paths or assumptions about locally saved teacher checkpoints and processed datasets. Update these paths before re-running the full training pipeline. The result-summary cells are included for inspection of the reported experimental outputs.
