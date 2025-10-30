<h1 align="center">IRT Scripts Documentation</h1>

## 1. Overview

This directory contains Python scripts that demonstrate Item Response Theory (IRT) concepts through simulation and visualization. These scripts are designed to be educational and showcase the practical application of IRT.

## 2. Scripts

### 2.1 `generate_irt_data.py`

This script generates synthetic IRT data for demonstration purposes.

#### What it does:
- Creates a set of item parameters (difficulty, discrimination, guessing)
- Generates student ability parameters from a normal distribution
- Simulates student responses using the 3PL IRT model
- Saves the generated data as CSV files in the `data/` directory

#### Usage:
```
python generate_irt_data.py
```

#### Output:
- `data/irt_responses.csv`: A matrix of student responses (1=correct, 0=incorrect)
- `data/irt_parameters.csv`: Item parameters for the generated items

### 2.2 `example_irt_simulation.py`

This script demonstrates core IRT concepts through simulation and visualization.

#### What it does:
1. Creates a visualization of Item Characteristic Curves (ICCs) showing how:
   - Difficulty (b) parameters shift the curve left/right
   - Discrimination (a) parameters affect the slope of the curve
   - Guessing (c) parameters set the lower asymptote

2. Simulates a simplified adaptive test showing:
   - How item selection works based on information theory
   - How ability estimates converge toward the true ability
   - The relationship between item difficulty and test-taker ability

#### Usage:
```
python example_irt_simulation.py
```

#### Output:
- `docs/item_characteristic_curves.png`: Visual representation of ICCs
- `docs/adaptive_testing_convergence.png`: Chart showing ability estimate convergence
- Console output showing the adaptive test sequence

#### Example Output:

The adaptive test simulation produces output similar to:

```
Adaptive Test Results:
   item_number  item_id  difficulty  response
0            1       11        0.00         1
1            2       15        1.58         1
2            3       18        2.53         0
3            4       13        0.79         1
4            5       16        1.89         1
5            6       19        2.84         0
6            7       17        2.21         0
7            8       14        1.26         1
8            9       12        0.42         1
9           10       10       -0.32         1
```

## 3. Implementation Logic

### 3.1 IRT Models

The scripts implement the following IRT models:

1. **1PL (Rasch) Model**:
   P(correct|θ) = 1 / (1 + e^(-(θ-b)))

2. **2PL Model**:
   P(correct|θ) = 1 / (1 + e^(-a(θ-b)))

3. **3PL Model**:
   P(correct|θ) = c + (1-c) / (1 + e^(-a(θ-b)))

Where:
- θ (theta) is the person's ability
- b is the item difficulty parameter
- a is the item discrimination parameter
- c is the guessing parameter

### 3.2 Item Information

The scripts calculate item information using:

I(θ) = a² × (P(θ) - c)² / ((1 - c)² × P(θ) × (1 - P(θ)))

Where:
- I(θ) is the information provided by the item at ability level θ
- P(θ) is the probability of a correct response at ability level θ

### 3.3 Adaptive Testing Logic

The adaptive testing simulation follows this algorithm:
1. Start with an initial ability estimate (typically θ = 0)
2. For each step until the stopping criterion:
   - Select the item with maximum information at current θ
   - Administer the item and get the response
   - Update the ability estimate
3. Final ability estimate is the result

The actual estimation procedure used here is simplified; operational CAT systems typically use more sophisticated approaches like Maximum Likelihood Estimation or Bayesian methods.
