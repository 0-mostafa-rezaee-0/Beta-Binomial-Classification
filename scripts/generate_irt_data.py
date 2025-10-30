#!/usr/bin/env python3
"""
Generate synthetic IRT data for demonstration purposes.
This script creates:
1. A set of item parameters (difficulty, discrimination, guessing)
2. A set of student ability parameters
3. A matrix of item responses based on the 3PL IRT model
"""

import numpy as np
import pandas as pd
import os

def generate_irt_data(n_students=500, n_items=20, model="3PL", seed=42):
    """
    Generate synthetic IRT data based on specified model.

    Parameters:
    -----------
    n_students : int
        Number of students to simulate
    n_items : int
        Number of items to simulate
    model : str
        IRT model to use ('1PL', '2PL', or '3PL')
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    responses_df : pandas DataFrame
        Matrix of student responses (1=correct, 0=incorrect)
    parameters_df : pandas DataFrame
        Item parameters for the generated items
    abilities : numpy array
        Generated ability parameters for students
    """
    np.random.seed(seed)

    # Generate student abilities from N(0, 1)
    abilities = np.random.normal(0, 1, size=n_students)

    # Generate item parameters
    difficulties = np.random.uniform(-3, 3, size=n_items)  # b parameters

    if model in ["2PL", "3PL"]:
        discriminations = np.random.uniform(0.5, 2.0, size=n_items)  # a parameters
    else:
        discriminations = np.ones(n_items)  # For 1PL, all discriminations = 1

    if model == "3PL":
        guessing = np.random.uniform(0.05, 0.25, size=n_items)  # c parameters
    else:
        guessing = np.zeros(n_items)  # For 1PL/2PL, no guessing

    # Generate response probabilities using the IRT model
    responses = np.zeros((n_students, n_items))
    for i in range(n_students):
        for j in range(n_items):
            theta = abilities[i]
            a = discriminations[j]
            b = difficulties[j]
            c = guessing[j]

            # 3PL model formula: P(correct) = c + (1-c) / (1 + exp(-a(theta-b)))
            p = c + (1 - c) / (1 + np.exp(-a * (theta - b)))
            responses[i, j] = np.random.binomial(1, p)

    # Create DataFrames
    responses_df = pd.DataFrame(
        responses,
        columns=[f"item_{j+1}" for j in range(n_items)]
    )
    responses_df.insert(0, "student_id", range(1, n_students + 1))

    parameters_df = pd.DataFrame({
        "item_id": [f"item_{j+1}" for j in range(n_items)],
        "difficulty": difficulties,
        "discrimination": discriminations,
        "guessing": guessing
    })

    return responses_df, parameters_df, abilities

def main():
    """Generate and save IRT data files."""
    # Make sure the data directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Generate data using the 3PL model
    print("Generating synthetic IRT data...")
    responses_df, parameters_df, _ = generate_irt_data(
        n_students=500,
        n_items=20,
        model="3PL"
    )

    # Save the data
    responses_path = os.path.join(data_dir, "irt_responses.csv")
    parameters_path = os.path.join(data_dir, "irt_parameters.csv")

    responses_df.to_csv(responses_path, index=False)
    parameters_df.to_csv(parameters_path, index=False)

    print(f"Data saved to '{data_dir}'")
    print(f"- Response matrix shape: {responses_df.shape}")
    print(f"- Parameters for {len(parameters_df)} items")

if __name__ == "__main__":
    main()
