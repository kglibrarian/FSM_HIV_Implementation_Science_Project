# HIV Implementation Science Metrics Project

## Overview

This repository contains code and documentation for analyzing HIV implementation science research publications from PubMed (2000-2025). The project uses a multi-stage data collection, validation, and analysis pipeline to identify, classify, and analyze trends in HIV implementation science literature with a focus on U.S.-based research.

**Special Note:** Text and code generated via ClaudeAI
**Institution:** Galter Health Sciences Library, Northwestern University Feinberg School of Medicine  
**Project Type:** Bibliometric analysis and research assessment  

---

## Table of Contents

- [Project Structure](#project-structure)
- [Methodology Overview](#methodology-overview)
- [Detailed Pipeline](#detailed-pipeline)
- [Code Organization](#code-organization)
- [Dependencies](#dependencies)
- [Usage Instructions](#usage-instructions)
- [Output Files](#output-files)
- [Data Quality](#data-quality)
- [Citation](#citation)

---

## Project Structure

```
project/
├── hiv not imp/                          # HIV WITHOUT implementation science
│   ├── pubmed_checkpoints/               # Incremental collection checkpoints
│   ├── hiv_us_2000_2025_FINAL.csv       # Raw dataset (~160k records)
│   ├── hiv_us_2000_2025_with_locations.csv
│   ├── hiv_locations_to_review.xlsx     # For manual curation
│   ├── hiv_locations_to_review_COMPLETED.xlsx
│   ├── hiv_us_2000_2025_with_locations_CLEANED.csv
│   ├── hiv_us_2000_2025_with_location_classification.csv
│   ├── location_hiv_cache.pkl           # Geopy cache
│   └── pubmed_errors.log
│
├── hiv imp/                              # HIV WITH implementation science
│   ├── pubmed_checkpoints/
│   ├── hiv_imp_us_2000_2025_FINAL.csv   # Raw dataset (~2,900 records)
│   ├── hiv_imp_us_2000_2025_with_locations.csv
│   ├── hiv_imp_locations_to_review.xlsx
│   ├── hiv_imp_locations_to_review_COMPLETED.xlsx
│   ├── hiv_imp_us_2000_2025_with_locations_CLEANED.csv
│   ├── hiv_imp_us_2000_2025_with_location_classification.csv
│   ├── location_hiv_imp_cache.pkl
│   └── pubmed_errors.log
│
├── hiv imp proj validation/              # ML validation outputs
│   ├── top_implementation_terms.csv
│   ├── potential_implementation_papers.xlsx
│   ├── validation_sample.xlsx
│   ├── validation_sample_COMPLETED.xlsx  # Your manual reviews
│   ├── file_a_with_scores.pkl
│   ├── hiv_us_2000_2025_with_location_and_impl_score.csv
│   ├── likely_implementation_papers.csv
│   └── filtered_2015-2025_with_DOI_likely_impl_US.csv  # FINAL DATASET
│
└── FSM_HIV_Implementation_Science_Metrics_Project.html  # Main notebook
```

---

## Methodology Overview

### Research Question
How has HIV implementation science evolved from 2000-2025, particularly in U.S.-based research? What are the publication trends, key authors, and funding patterns?

### Approach
1. **Dual search strategy:** Collect both HIV implementation science AND HIV non-implementation publications
2. **Geographic validation:** Use NLP + manual curation to identify U.S.-focused research
3. **ML validation:** Use machine learning to identify missed implementation science papers
4. **Trend analysis:** Analyze growth rates, authorship patterns, and funding

---

## Detailed Pipeline

### Stage 1: PubMed Data Collection

**Two complementary searches:**

#### Search A: HIV AND Implementation Science (~2,900 records)
```
(HIV terms in title OR HIV journals) 
AND (Implementation science MeSH terms OR keywords)
AND English[lang] 
AND 2000-2025
NOT (non-U.S. geographic terms)
```

**Implementation science terms include:**
- MeSH: "Implementation Science"[majr]
- Keywords: implement*, delivery-science, dissemination-science, barriers, facilitator*, program-evaluation
- Concepts: scale-up, real-world, pilot, workflow, training, fidelity, adoption, sustainability

#### Search B: HIV NOT Implementation Science (~160,800 records)
```
(HIV terms in title OR HIV journals)
NOT (Implementation science terms)
AND English[lang]
AND 2000-2025
NOT (non-U.S. geographic terms)
```

**Technical implementation:**
- **Date chunking:** 51 chunks (6-month intervals) to avoid API timeouts
- **Checkpoint system:** Saves progress every 50 batches
- **Rate limiting:** 10 requests/second with API key
- **Affiliation tracking:** Flags Northwestern-affiliated authors
- **Resumable:** Can restart from last checkpoint if interrupted

**Key functions:**
- `fetch_pubmed_record()` - Retrieve individual records
- `process_pubmed_record()` - Extract metadata (title, abstract, authors, DOI, etc.)
- `get_authors_with_affiliation_name_full()` - Track institutional affiliations

---

### Stage 2: Geographic Classification (NLP + Manual Curation)

**Goal:** Identify which papers focus on U.S. settings vs. international settings

#### Step 2.1: Location Extraction (spaCy)
- Uses spaCy `en_core_web_sm` model for Named Entity Recognition (NER)
- Extracts GPE (Geopolitical Entity) tags from titles + abstracts
- Processes in 10k record batches for memory efficiency
- Output: List of all location mentions per paper

#### Step 2.2: Manual Curation
**Problem:** NLP produces false positives (drug names, acronyms, diseases misclassified as places)

**Solution:**
1. Export unique locations to Excel (`hiv_locations_to_review.xlsx`)
2. Manual review: Mark false positives with 'X' in DELETE? column
3. Examples of false positives: "AIDS" (disease, not location), "HAART" (treatment), "CD4" (marker)

#### Step 2.3: Location Cleaning
- Read `hiv_locations_to_review_COMPLETED.xlsx`
- Remove all locations marked for deletion
- Output: `hiv_us_2000_2025_with_locations_CLEANED.csv`

#### Step 2.4: U.S. Classification (Hybrid Approach)
**Three-stage classification:**

**Stage 1 - Rule-based (instant):**
```python
us_locations = {
    'United States', 'USA', 'US', 'U.S.', 'America',
    # All 50 states (full names + abbreviations)
    'California', 'CA', 'New York', 'NY', 'Texas', 'TX', ...
    # Territories
    'Puerto Rico', 'Guam', 'Virgin Islands', 'American Samoa',
    # Major cities
    'New York City', 'Los Angeles', 'Chicago', 'Houston', ...
    # Regions
    'Midwest', 'Northeast', 'Pacific Northwest', 'New England'
}
```

**Stage 2 - Pattern matching (instant):**
```python
us_patterns = [
    r'\bCounty\b', r'\bParish\b', r'\bBorough\b',
    r'\bMetropolitan\b', r'\bMetro\b', r'\bGreater\s+\w+'
]
```

**Stage 3 - Geopy lookup (slow, only for unknowns):**
- Uses Nominatim geocoder from OpenStreetMap
- 1.1 second rate limit per request
- Results cached in `location_hiv_cache.pkl`
- User prompted before running (shows estimated time)

**Final categories:**
- `US only` - All locations are U.S.-based
- `Non-US only` - All locations are non-U.S.
- `Mixed (US + Non-US)` - Contains both
- `No locations` - No geographic data extracted

---

### Stage 3: Implementation Science Validation (Machine Learning)

**Problem:** Keyword searches miss papers that discuss implementation concepts but don't use standard terminology

**Example:** A paper about "scaling up HIV treatment in community clinics" might not use "implementation science" but is clearly implementation-focused.

#### Step 3.1: TF-IDF Training
```python
# Train on known implementation science papers (File B)
vectorizer = TfidfVectorizer(
    max_features=5000,      # Top 5000 terms
    ngram_range=(1, 3),     # 1-3 word phrases
    min_df=5,               # Must appear in 5+ papers
    stop_words='english'
)
tfidf_implementation = vectorizer.fit_transform(file_b['text'])
```

**Extracts characteristic terms like:**
- "scale up", "real world implementation", "clinic level"
- "barriers to implementation", "fidelity assessment"
- "pilot program", "pragmatic trial", "workflow integration"

#### Step 3.2: Similarity Scoring
```python
# Score all non-implementation papers (File A)
implementation_centroid = tfidf_implementation.mean(axis=0)
similarities = cosine_similarity(tfidf_file_a, implementation_centroid)
```

Each paper gets a score from 0.0 to 1.0:
- **0.0-0.1:** Very unlikely to be implementation science
- **0.3-0.5:** Moderate similarity
- **0.7-1.0:** Very likely implementation science

#### Step 3.3: Validation Sampling
**Stratified sampling across similarity bins:**
```python
bins = [0, 0.1, 0.2, 0.3, 0.4, 1.0]
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
# Sample 50 papers from each bin
```

**Manual review process:**
1. Reviewer reads title + abstract
2. Marks 'Y' if paper discusses implementation concepts
3. Marks 'N' if paper is basic science/clinical trial only

#### Step 3.4: Threshold Determination
```python
# Find minimum similarity score among papers marked 'Y'
threshold = implementation_papers['implementation_similarity'].min()

# Apply to full dataset
file_a['likely_implementation'] = file_a['implementation_similarity'] >= threshold
```

#### Step 3.5: Merge and Export
- Merge similarity scores to full dataset with location classification
- Flag papers as `likely_implementation` = True/False
- Export for final analysis

---

### Stage 4: Final Dataset Construction

**Filtering criteria for analysis dataset:**
```python
filtered_df = df[
    (df['date_year'] >= 2015) &          # Recent decade
    (df['date_year'] <= 2025) &
    (df['DOI'].notna()) &                 # Has DOI
    (df['DOI'].str.strip() != '') &
    (df['likely_implementation'] == True) &  # Implementation science
    (df['Location_Detail'] != 'Non-US only')  # U.S.-related
]
```

**Final output:** `filtered_2015-2025_with_DOI_likely_impl_US.csv`

This is the dataset used for all analyses.

---

## Code Organization

### Main Analysis Notebook
`FSM_HIV_Implementation_Science_Metrics_Project.html`

**Contains:**
1. **Function definitions:** PubMed data extraction utilities
2. **Data collection:** Chunked PubMed retrieval
3. **Location processing:** spaCy extraction and classification
4. **ML validation:** TF-IDF training and scoring
5. **Trend analysis:** Growth rates, author metrics
6. **Network analysis:** Co-authorship patterns
7. **Funding analysis:** Top funders and trends

### Key Functions

#### Data Collection
```python
fetch_pubmed_record(pubmed_id)
# Retrieves XML record from PubMed

process_pubmed_record(record, affiliations_to_check)
# Extracts: PMID, DOI, title, abstract, authors, journal, date, 
#           publication type, Northwestern affiliation

get_authors_with_affiliation_name_full(record, affiliations)
# Returns list of authors with specified affiliation
```

#### Location Processing
```python
extract_locations_spacy(text)
# Uses spaCy NER to extract GPE entities

is_us_location_hybrid(location, cache)
# Three-stage U.S. classification: rules → patterns → geopy
```

#### ML Validation
```python
TfidfVectorizer(max_features=5000, ngram_range=(1,3), min_df=5)
# Creates term-frequency vectors

cosine_similarity(vectors_a, centroid_b)
# Calculates similarity scores
```

---

## Dependencies

### Required Python Packages
```bash
pip install pandas numpy biopython spacy scikit-learn geopy openpyxl
python -m spacy download en_core_web_sm
```

### Package Versions (Recommended)
```
pandas >= 1.5.0
numpy >= 1.23.0
biopython >= 1.81
spacy >= 3.5.0
scikit-learn >= 1.2.0
geopy >= 2.3.0
openpyxl >= 3.1.0  # For Excel file handling
```

### Optional (for visualizations)
```bash
pip install matplotlib seaborn plotly networkx
```

---

## Usage Instructions

### 1. Data Collection

```python
# Configure
INPUT_FOLDER = 'hiv imp'  # or 'hiv not imp'
Entrez.email = "your.email@institution.edu"
Entrez.api_key = "your_api_key"

# Run collection
# Automatically creates checkpoints and handles interruptions
# Takes 4-6 hours for full dataset
```

**To restart from checkpoint:**
- Script automatically detects existing chunk files
- Skips already-downloaded chunks
- Resumes from last incomplete chunk

**To start fresh:**
```python
import shutil
checkpoint_dir = os.path.join(INPUT_FOLDER, 'pubmed_checkpoints')
shutil.rmtree(checkpoint_dir)
```

### 2. Location Processing

**Step A: Extract locations**
```python
INPUT_FOLDER = 'hiv not imp'  # or 'hiv imp'
# Runs spaCy NER on title + abstract
# Output: hiv_us_2000_2025_with_locations.csv
```

**Step B: Review locations**
```python
# Exports: hiv_locations_to_review.xlsx
# Open in Excel, mark false positives with 'X' in DELETE? column
# Save as: hiv_locations_to_review_COMPLETED.xlsx
```

**Step C: Clean locations**
```python
# Reads COMPLETED file
# Removes marked locations
# Output: hiv_us_2000_2025_with_locations_CLEANED.csv
```

**Step D: Classify U.S. locations**
```python
# Hybrid classification (rules + patterns + geopy)
# Creates cache: location_hiv_cache.pkl
# Output: hiv_us_2000_2025_with_location_classification.csv
```

### 3. ML Validation

**Step A: Train and score**
```python
OUTPUT_FOLDER = 'hiv imp proj validation'
FILE_A_FOLDER = 'hiv not imp'  # Non-implementation
FILE_B_FOLDER = 'hiv imp'      # Implementation

# Trains TF-IDF on File B
# Scores all File A papers
# Outputs:
#   - top_implementation_terms.csv
#   - potential_implementation_papers.xlsx
#   - validation_sample.xlsx
```

**Step B: Manual validation**
```python
# Open: validation_sample.xlsx
# Review 250 papers
# Mark 'Y' or 'N' in YOUR_REVIEW column
# Save as: validation_sample_COMPLETED.xlsx
```

**Step C: Apply threshold**
```python
# Reads COMPLETED file
# Determines optimal threshold
# Flags likely implementation papers
# Outputs:
#   - hiv_us_2000_2025_with_location_and_impl_score.csv
#   - likely_implementation_papers.csv
```

### 4. Final Filtering

```python
# Apply final filters
#   - Years: 2015-2025
#   - Has DOI: Yes
#   - Likely implementation: True
#   - Location: NOT "Non-US only"
# Output: filtered_2015-2025_with_DOI_likely_impl_US.csv
```

### 5. Run Analyses

Use the final filtered dataset for:
- Publication growth trends
- Author analyses (new authors, prolific authors)
- Network analysis (co-authorship, keywords)
- Funding analysis

---

## Output Files

### Datasets

| File | Description | Records |
|------|-------------|---------|
| `hiv_us_2000_2025_FINAL.csv` | HIV non-implementation raw | ~160k |
| `hiv_imp_us_2000_2025_FINAL.csv` | HIV implementation raw | ~2.9k |
| `hiv_us_2000_2025_with_location_classification.csv` | With U.S. classification | ~160k |
| `hiv_us_2000_2025_with_location_and_impl_score.csv` | With ML scores | ~160k |
| `filtered_2015-2025_with_DOI_likely_impl_US.csv` | **FINAL ANALYSIS DATASET** | Variable |

### Validation Files

| File | Purpose |
|------|---------|
| `top_implementation_terms.csv` | Top 100 TF-IDF terms for implementation science |
| `validation_sample.xlsx` | 250 papers for manual review |
| `validation_sample_COMPLETED.xlsx` | Your completed reviews |

### Cache Files

| File | Purpose |
|------|---------|
| `location_hiv_cache.pkl` | Geopy geocoding results (speeds up re-runs) |
| `location_hiv_imp_cache.pkl` | Cache for implementation dataset |
| `file_a_with_scores.pkl` | Saved scored dataset |

---

## Key Metrics & Analyses

### 1. Publication Growth Analysis

**Year-over-Year (YoY) Growth:**
```python
YoY_Growth = ((Count_CurrentYear - Count_PreviousYear) / Count_PreviousYear) × 100
```

**Compound Annual Growth Rate (CAGR):**
```python
CAGR = ((Final_Year / First_Year)^(1/Years) - 1) × 100
```

**Implementation Science as % of Total HIV:**
```python
Impl_Pct = (HIV_Impl_Count / Total_HIV_Count) × 100
```

### 2. Author Analyses

**New Authors:** First-time contributors per year
```python
# Authors not in previous years' publications
```

**Prolific Authors:** Researchers with 3+ publications
```python
author_counts = df.groupby('Author').size()
prolific = author_counts[author_counts >= 3]
```

### 3. Network Analysis

**Exports for VOSviewer:**
- Co-authorship networks
- Keyword co-occurrence
- Citation networks

**Highlighting implementation science subset within broader HIV network**

### 4. Funding Analysis

**Top funders:**
```python
# Extract from Grant acknowledgments
# Rank by frequency and funding amount
```

---

## Data Quality & Limitations

### Strengths
✅ **Comprehensive coverage:** 2000-2025, 160k+ HIV papers  
✅ **Validated classification:** Manual review of 250+ papers  
✅ **Reproducible:** Checkpoint system, cached results  
✅ **Multi-stage validation:** NLP + manual curation + ML  
✅ **Institutional tracking:** Northwestern affiliation flagged  

### Limitations
⚠️ **Geographic classification:** ~90-95% accuracy (some ambiguous cases)  
⚠️ **Implementation detection:** Dependent on author terminology  
⚠️ **Training set bias:** ML model limited by keyword search results  
⚠️ **PubMed coverage:** May miss gray literature, preprints  
⚠️ **Affiliation quality:** Varies by journal and time period  

### Quality Assurance Steps
1. ✅ Deduplication by PMID
2. ✅ Manual curation of 3,000+ location names
3. ✅ Validation sample of 250 papers reviewed
4. ✅ Checkpoint system prevents data loss
5. ✅ Error logging throughout pipeline
6. ✅ Cache system for reproducibility

---

## Troubleshooting

### Common Issues

**Issue: API timeouts**
```python
# Solution: Checkpoint system automatically resumes
# Check: pubmed_errors.log for details
```

**Issue: geopy rate limiting**
```python
# Solution: Results are cached
# Rerun uses cache, doesn't re-query
```

**Issue: Memory errors during spaCy**
```python
# Solution: Process in 10k record chunks
# Adjust: chunk_size parameter if needed
```

**Issue: Missing checkpoint directory**
```python
# Solution: Script auto-creates on first run
# Manually create: os.makedirs('hiv imp/pubmed_checkpoints')
```

---

## Reproducibility Checklist

- [ ] PubMed API key configured
- [ ] Email address set in Entrez.email
- [ ] spaCy `en_core_web_sm` model downloaded
- [ ] All dependencies installed
- [ ] Folder structure created (`hiv imp/`, `hiv not imp/`, validation folder)
- [ ] Manual review of locations completed
- [ ] Manual validation of 250 papers completed
- [ ] Threshold applied and final dataset created

---

## Version History

### v1.0 (December 2024)
- Initial data collection (2000-2025)
- Geographic classification with NLP + manual curation
- ML-based implementation science detection
- Publication growth analysis
- Author trend analysis
- Network analysis preparation
- Funding landscape analysis

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@misc{gutzman2024hiv,
  title={HIV Implementation Science Metrics Project: 
         A Machine Learning Approach to Identifying Implementation Research},
  author={Gutzman, Karen},
  year={2024},
  institution={Northwestern University Feinberg School of Medicine, 
               Galter Health Sciences Library},
  howpublished={\url{https://github.com/[repository-url]}}
}
```

---

## Contact & Support

**Karen Gutzman**  
Research Data & Assessment Librarian  
Galter Health Sciences Library  
Northwestern University Feinberg School of Medicine  
📧 karen.gutzman@northwestern.edu

**For questions about:**
- Methodology: Contact Karen
- Code issues: Open GitHub issue
- Data access: Contact Northwestern Research Services

---

## Acknowledgments

- **ClaudeAI** 
- **Northwestern University Feinberg School of Medicine**
- **Galter Health Sciences Library**
- **NCBI E-utilities** (PubMed API)
- **spaCy NLP library**
- **OpenStreetMap/Nominatim** (geocoding)
- **scikit-learn** (machine learning)
- **Project collaborators:** Mao Soulakis, Corinne Miller, Nanette Benbow, and the research team

---

## License

[Specify license - e.g., MIT, CC-BY-4.0, or institutional license]

---

## Appendix: File Naming Conventions

### Non-implementation folder (`hiv not imp/`)
```
hiv_us_2000_2025_FINAL.csv
hiv_us_2000_2025_with_locations.csv
hiv_us_2000_2025_with_locations_CLEANED.csv
hiv_us_2000_2025_with_location_classification.csv
hiv_us_2000_2025_with_location_and_impl_score.csv
hiv_locations_to_review.xlsx
hiv_locations_to_review_COMPLETED.xlsx
location_hiv_cache.pkl
pubmed_errors.log
```

### Implementation folder (`hiv imp/`)
```
hiv_imp_us_2000_2025_FINAL.csv
hiv_imp_us_2000_2025_with_locations.csv
hiv_imp_us_2000_2025_with_locations_CLEANED.csv
hiv_imp_us_2000_2025_with_location_classification.csv
hiv_imp_locations_to_review.xlsx
hiv_imp_locations_to_review_COMPLETED.xlsx
location_hiv_imp_cache.pkl
pubmed_errors.log
```

### Validation folder (`hiv imp proj validation/`)
```
top_implementation_terms.csv
potential_implementation_papers.xlsx
validation_sample.xlsx
validation_sample_COMPLETED.xlsx
file_a_with_scores.pkl
hiv_us_2000_2025_with_location_and_impl_score.csv
likely_implementation_papers.csv
filtered_2015-2025_with_DOI_likely_impl_US.csv
publication_growth_analysis.csv (if generated)
```

---

**Last Updated:** December 2024  
**Version:** 1.0  
**Status:** Active Development
