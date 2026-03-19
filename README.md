# FSM HIV Implementation Science Metrics Project

Bibliometric pipeline for identifying and analyzing domestic HIV implementation science (IS) literature from PubMed (2000–2025), with a focus on U.S.-based research.

**Institution:** Galter Health Sciences Library, Northwestern University Feinberg School of Medicine
**Status:** Active Development | **Version:** 2.0 (March 2026)

> Code developed with assistance from Claude AI (Anthropic)

---

## What This Pipeline Does

1. **Fetches** two complementary PubMed datasets — HIV IS papers and HIV non-IS papers — using structured Boolean queries
2. **Classifies** each paper geographically using spaCy NER + an authoritative jurisdiction term list
3. **Scores** non-IS papers using TF-IDF cosine similarity against the IS corpus to identify papers missed by the keyword search
4. **Validates** results through gold standard matching and optional researcher manual review
5. **Produces** two final publication-ready datasets with full provenance columns

---

## Project Structure

```
project/
├── hiv imp/                          # HIV AND IS keyword query outputs
│   ├── chunks/                       # Intermediate fetch chunks
│   ├── pubmed_checkpoints/           # Resumable fetch progress
│   ├── spacy_checkpoints/            # Resumable spaCy progress
│   ├── hiv_imp_us_2000_2025_FINAL.csv
│   ├── hiv_imp_us_2000_2025_with_locations.csv
│   └── pubmed_errors.log
│
├── hiv not imp/                      # HIV NOT IS keyword query outputs
│   ├── chunks/
│   ├── pubmed_checkpoints/
│   ├── spacy_checkpoints/
│   ├── hiv_us_2000_2025_FINAL.csv
│   ├── hiv_us_2000_2025_with_locations.csv
│   └── pubmed_errors.log
│
├── data/                             # Reference files
│   ├── 2026_03-12_List_of_Domestic_Foreign_Jurisdictions.xlsx
│   ├── 2026_03-12_Basic_science_terms.xlsx
│   └── gold_standard_is_papers.csv
│
└── outputs/
    ├── 07_location_rules/            # Cell 7: spaCy-based location flags
    ├── 08_location_jurisdictions/    # Cell 8: authoritative location flags
    ├── 09_ml_scoring/               # Cell 9: TF-IDF similarity scores
    ├── 10_gold_standard/            # Cell 10: gold standard with PMIDs
    ├── 11_threshold/                # Cell 11: threshold calibration
    ├── 12_exclusion_flags/          # Cell 12: hiv not imp exclusion flags
    ├── 13_hiv_imp_qc/               # Cell 13: hiv imp QC flags
    ├── 14_combined/                 # Cell 14: combined IS dataset
    ├── 15_review_sample/            # Cell 15: 10% manual review sample
    ├── 16_review_applied/           # Cell 16: post-review dataset
    └── 17_final_datasets/           # Cell 17: publication-ready outputs
```

---

## Pipeline Cells

| Cell | Purpose |
|------|---------|
| 1 | Import libraries |
| 2 | Project paths and `ACTIVE_DATASET` switch |
| 3 | Checkpoint manager (fresh start or resume) |
| 4 | PubMed record parsing functions |
| 5 | PubMed fetch loop (chunked, resumable) |
| 6 | spaCy geographic entity extraction |
| 7 | Geographic classification: rule-based |
| 8 | Geographic classification: jurisdiction lists |
| 9 | ML similarity scoring (TF-IDF) |
| 10 | Gold standard PMID lookup — run once |
| 11 | Similarity threshold calibration |
| 12 | IS exclusion flags — hiv not imp |
| 13 | QC flags — hiv imp |
| 14 | Combine both IS sources |
| 15 | Stratified 10% manual review sample |
| 16 | Apply manual review decisions |
| 17 | Final datasets |

---

## Run Order

**Cells 1–8** must be run once per dataset. Switch `ACTIVE_DATASET` in Cell 2 between runs.

```
# First run — hiv not imp
ACTIVE_DATASET = FOLDER_HIV_NOT_IMP
Run: Cells 1, 2, 3, 4, 5, 6, 7, 8

# Second run — hiv imp
ACTIVE_DATASET = FOLDER_HIV_IMP
Re-run Cell 2, then: Cells 3, 5, 6, 7, 8

# Remainder — no switch needed
Run: Cells 9, 10, 11, 12, 13, 14
Run: Cell 17 → preliminary datasets (optional, for early review)
Run: Cell 15 → send manual_review_sample.xlsx to researchers
Run: Cells 16, 17 → final datasets after review is returned
```

See the **Run Order and Researcher Handoff Guide** in the first notebook cell for full instructions including researcher handoff points.

---

## Search Strategy

Two PubMed queries, both scoped to English-language, 2000–2025, U.S.-related papers:

| Dataset | Logic | ~Records |
|---------|-------|---------|
| hiv imp | `HIV_BASE AND IS_terms AND date NOT foreign-only` | ~3,900 |
| hiv not imp | `HIV_BASE AND date NOT foreign-only NOT IS_terms` | ~160,000 |

Geographic filtering uses a two-stage approach: PubMed query excludes papers mentioning only foreign countries (with a US rescue clause), and Cell 8 applies the NU FSM Impact Institute's comprehensive jurisdiction term lists post-hoc.

Query version history is preserved in Cell 5. The active queries are `QUERY_HIV_IMP` and `QUERY_HIV_NOT_IMP` (Version 2, March 2026).

---

## Dependencies

```bash
pip install pandas numpy biopython spacy scikit-learn openpyxl joblib
python -m spacy download en_core_web_sm
```

| Package | Purpose |
|---------|---------|
| biopython | PubMed Entrez API |
| spacy | Geographic entity extraction |
| scikit-learn | TF-IDF similarity scoring |
| openpyxl | Excel file handling |
| joblib | Saving fitted TF-IDF vectorizer |

---

## Final Outputs

Located in `outputs/17_final_datasets/`:

| File | Description |
|------|-------------|
| `final_hiv_imp_IS_dataset.csv` | Confirmed domestic HIV IS papers |
| `final_hiv_not_imp_dataset.csv` | HIV non-IS comparison dataset |
| `final_datasets_summary.txt` | Record counts and column inventory |

Both files retain all pipeline columns (similarity scores, exclusion flags, location classifications, gold standard indicators, review decisions) for methods documentation.

---

## Key Design Decisions

- **`[ti]` not `[tiab]`** for IS search terms — abstract-level expansion tested and found to add ~21,000 non-IS papers with negligible IS recall improvement
- **Two-stage geography** — PubMed query handles obvious exclusions; Cell 8 jurisdiction lists catch terms spaCy misses (LMICs, sub-Saharan Africa, etc.)
- **PEPFAR/USAID override** — papers mentioning these programs are reclassified as international regardless of spaCy output
- **Three-tier threshold calibration** — researcher validation > gold standard PMID matching > hard default (0.15)
- **All pipeline columns preserved** in final outputs for methods reproducibility

---

## Limitations

- Geographic classification cannot catch papers about international settings that mention no country name in the title or abstract
- ML scoring is trained on the IS keyword corpus, so it inherits any biases in that query
- PubMed `pdat` (publication date) filtering may miss papers published within range but indexed later
- spaCy `en_core_web_sm` extracts GPE entities but misses many location references that appear only as adjectives or regional terms

---

## Citation

```bibtex
@misc{gutzman2026hiv,
  title={FSM HIV Implementation Science Metrics Project},
  author={Gutzman, Karen and Mustanski, Brian and Delehant, Molly and
          Benbow, Nanette Dior and Li, Dennis H. and Lowther, Matthew and
          Soulakis, Mao},
  year={2026},
  institution={Northwestern University Feinberg School of Medicine}
}
```

---

## Contact & Authors

**Karen Gutzman** — Research Data & Assessment Librarian
Galter Health Sciences Library, Northwestern University Feinberg School of Medicine
karen.gutzman@northwestern.edu

**Project Team:**

| Name | Institution | Contact |
|------|-------------|---------|
| Karen Gutzman | Galter Health Sciences Library, Northwestern | karen.gutzman@northwestern.edu |
| Brian Mustanski | Northwestern University | brian@northwestern.edu |
| Molly Delehant | Northwestern University | molly.delehant@northwestern.edu |
| Nanette Dior Benbow | Northwestern University | nanette@northwestern.edu |
| Dennis H. Li | Northwestern University | dennis@northwestern.edu |
| Matthew Lowther | Northwestern University | matthew.lowther@northwestern.edu |
| Mao Soulakis | Northwestern University | mao.soulakis@northwestern.edu |

**Additional collaborators:** Corinne Miller, Amy, Christina, and the ISCI Team

---

## Acknowledgments

[NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25499/) · [spaCy](https://spacy.io) · [scikit-learn](https://scikit-learn.org) · [Claude AI](https://www.anthropic.com) (Anthropic)

