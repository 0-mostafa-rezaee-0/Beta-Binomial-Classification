<h1 align="center">IRT Dataset Information</h1>

## 1. Dataset Overview

For this project, we're using the publicly available PISA 2015 dataset, specifically a subset focused on mathematics items. The Programme for International Student Assessment (PISA) is a worldwide study by the OECD that evaluates educational systems by measuring 15-year-old students' scholastic performance in mathematics, science, and reading.

## 2. Dataset Access

### 2.1 Download Link
You can access the full PISA 2015 dataset from the official OECD website:
https://www.oecd.org/pisa/data/2015database/

For this project, we recommend using a simplified subset that contains:
- Student responses to mathematics items
- Item parameters
- Student ability estimates

### 2.2 Simplified Dataset
To simplify your experience, we've included a script that will generate a synthetic dataset based on IRT principles. This allows you to understand the structure of IRT data without having to download and process the large PISA dataset.

Run the data generation script from the repository root:
```
python scripts/generate_irt_data.py
```

This will create two files in the data directory:
- `irt_responses.csv`: A matrix of student responses (1=correct, 0=incorrect)
- `irt_parameters.csv`: Item parameters for the generated items

## 3. Data Structure

### 3.1 Response Data (`irt_responses.csv`)

The response data is structured as follows:

| student_id | item_1 | item_2 | ... | item_20 |
|------------|--------|--------|-----|---------|
| 1          | 1      | 0      | ... | 1       |
| 2          | 0      | 0      | ... | 0       |
| ...        | ...    | ...    | ... | ...     |
| 500        | 1      | 1      | ... | 0       |

Where:
- `student_id`: Unique identifier for each student
- `item_X`: Binary response (1=correct, 0=incorrect) for item X

### 3.2 Item Parameters (`irt_parameters.csv`)

The item parameters file contains:

| item_id | difficulty | discrimination | guessing |
|---------|------------|----------------|----------|
| item_1  | -1.2       | 1.1            | 0.2      |
| item_2  | 0.5        | 0.9            | 0.1      |
| ...     | ...        | ...            | ...      |
| item_20 | 2.3        | 1.5            | 0.05     |

Where:
- `item_id`: Unique identifier for each item
- `difficulty`: Item difficulty parameter (b) on the logit scale
- `discrimination`: Item discrimination parameter (a) 
- `guessing`: Lower asymptote parameter (c) for 3PL model

## 4. Using This Data for IRT Modeling

This dataset can be used to:

1. **Estimate item parameters** if they are unknown
2. **Estimate student abilities** based on their response patterns
3. **Simulate adaptive testing** by selecting items based on current ability estimates
4. **Visualize item characteristic curves** to understand item behavior
5. **Calculate test information** to assess measurement precision across the ability spectrum

The example scripts in this repository demonstrate several of these applications.

## 5. Note on Real-world Applications

In operational settings, IRT models are typically estimated on large samples (1000+ respondents) with specialized software packages. The examples in this repository are simplified for educational purposes but follow the same principles used in high-stakes testing programs like GRE, GMAT, and PTE.
