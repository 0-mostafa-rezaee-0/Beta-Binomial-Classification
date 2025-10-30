#!/usr/bin/env python3
"""
Example IRT Simulation
---------------------
This script demonstrates basic IRT concepts by:
1. Creating items with different difficulty parameters
2. Simulating users with different ability levels
3. Computing probability of correct response
4. Visualizing item characteristic curves
5. Demonstrating a simple adaptive testing sequence
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def irt_probability(theta, b, a=1.0, c=0.0):
    """
    Calculate the probability of a correct response using IRT models.

    Parameters:
    -----------
    theta : float or array
        Ability parameter(s)
    b : float
        Item difficulty parameter
    a : float, optional
        Item discrimination parameter (default=1.0 for 1PL model)
    c : float, optional
        Item guessing parameter (default=0.0 for 1PL/2PL models)

    Returns:
    --------
    p : float or array
        Probability of correct response
    """
    # 3PL model: P(correct) = c + (1-c) / (1 + exp(-a(theta-b)))
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))

def item_information(theta, b, a=1.0, c=0.0):
    """
    Calculate the item information function.

    Parameters:
    -----------
    theta : float or array
        Ability parameter(s)
    b, a, c : float
        Item parameters

    Returns:
    --------
    info : float or array
        Item information at given ability level(s)
    """
    p = irt_probability(theta, b, a, c)
    q = 1 - p
    return (a**2) * ((p - c)**2) / ((1 - c)**2) * (q / p)

def simulate_response(theta, b, a=1.0, c=0.0):
    """
    Simulate a response to an item based on IRT probability.

    Returns:
    --------
    response : int
        1 for correct, 0 for incorrect
    """
    p = irt_probability(theta, b, a, c)
    return np.random.binomial(1, p)

def plot_item_characteristic_curves():
    """Plot item characteristic curves for items with different parameters."""
    # Ensure the output directory exists
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(os.path.dirname(script_dir), "figures")
    os.makedirs(figures_dir, exist_ok=True)

    theta_range = np.linspace(-4, 4, 100)

    # Create figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Effect of difficulty (b) parameter
    ax = axes[0]
    difficulty_levels = [-2, -1, 0, 1, 2]
    for b in difficulty_levels:
        p = irt_probability(theta_range, b=b)
        ax.plot(theta_range, p, label=f'b = {b}')

    ax.set_title('Effect of Difficulty Parameter (1PL Model)')
    ax.set_xlabel('Ability (θ)')
    ax.set_ylabel('Probability of Correct Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Effect of discrimination (a) parameter
    ax = axes[1]
    discrimination_levels = [0.5, 1.0, 1.5, 2.0]
    for a in discrimination_levels:
        p = irt_probability(theta_range, b=0, a=a)
        ax.plot(theta_range, p, label=f'a = {a}')

    ax.set_title('Effect of Discrimination Parameter (2PL Model)')
    ax.set_xlabel('Ability (θ)')
    ax.set_ylabel('Probability of Correct Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Effect of guessing (c) parameter
    ax = axes[2]
    guessing_levels = [0.0, 0.1, 0.2, 0.25]
    for c in guessing_levels:
        p = irt_probability(theta_range, b=0, a=1, c=c)
        ax.plot(theta_range, p, label=f'c = {c}')

    ax.set_title('Effect of Guessing Parameter (3PL Model)')
    ax.set_xlabel('Ability (θ)')
    ax.set_ylabel('Probability of Correct Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'item_characteristic_curves.png'))
    plt.close()

def simulate_adaptive_test():
    """
    Simulate a simple adaptive test for a student.

    This demonstrates the basic principle of adaptive testing:
    1. Start with an initial ability estimate
    2. Select the most informative item
    3. Administer the item and get response
    4. Update ability estimate
    5. Repeat until stopping criterion is met
    """
    # True ability of the simulated student
    true_ability = 1.2

    # Create a pool of items with known parameters
    item_pool = pd.DataFrame({
        'item_id': range(1, 21),
        'b': np.linspace(-3, 3, 20),  # Difficulties from -3 to 3
        'a': np.random.uniform(0.7, 1.8, 20),  # Discriminations
        'c': np.random.uniform(0.0, 0.25, 20)   # Guessing parameters
    })

    # Initialize test
    max_items = 10
    current_estimate = 0.0  # Initial ability estimate (middle of scale)
    administered_items = []
    responses = []
    ability_estimates = [current_estimate]

    # Run adaptive algorithm
    for i in range(max_items):
        # Find available items (not yet administered)
        available_items = item_pool[~item_pool['item_id'].isin(administered_items)]

        # Calculate information for all available items at current ability estimate
        # Create a copy to avoid the SettingWithCopyWarning
        available_items = available_items.copy()
        available_items.loc[:, 'info'] = available_items.apply(
            lambda row: item_information(current_estimate, row['b'], row['a'], row['c']), 
            axis=1
        )

        # Select most informative item
        next_item = available_items.loc[available_items['info'].idxmax()]

        # Simulate response
        response = simulate_response(
            true_ability,
            next_item['b'],
            next_item['a'],
            next_item['c']
        )

        # Update administered items and responses
        administered_items.append(next_item['item_id'])
        responses.append(response)

        # Update ability estimate (simplified - in practice would use MLE or Bayesian methods)
        # This is a simplified estimate update using response pattern
        if response == 1:  # Correct
            current_estimate = current_estimate + (1 - irt_probability(
                current_estimate, next_item['b'], next_item['a'], next_item['c']
            )) * 0.5
        else:  # Incorrect
            current_estimate = current_estimate - irt_probability(
                current_estimate, next_item['b'], next_item['a'], next_item['c']
            ) * 0.5

        ability_estimates.append(current_estimate)

    # Create results DataFrame
    results = pd.DataFrame({
        'item_number': range(1, max_items + 1),
        'item_id': administered_items,
        'difficulty': [item_pool.loc[item_pool['item_id'] == item_id, 'b'].values[0] 
                      for item_id in administered_items],
        'response': responses
    })

    # Plot ability estimate convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(max_items + 1), ability_estimates, 'bo-')
    plt.axhline(y=true_ability, color='r', linestyle='--',
                label=f'True Ability = {true_ability}')
    plt.xlabel('Number of Items Administered')
    plt.ylabel('Ability Estimate (θ)')
    plt.title('Adaptive Testing: Ability Estimate Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Ensure the output directory exists
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(os.path.dirname(script_dir), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'adaptive_testing_convergence.png'))
    plt.close()

    return results

def main():
    """Run IRT simulations and visualizations."""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(os.path.dirname(script_dir), "figures")
    print("Running IRT simulation examples...")

    # 1. Plot item characteristic curves
    plot_item_characteristic_curves()
    print("- Generated item characteristic curves")

    # 2. Simulate a simple adaptive test
    results = simulate_adaptive_test()
    print("- Simulated adaptive test")
    print("\nAdaptive Test Results:")
    print(results)

    print(f"\nSimulation complete! Visualization images saved to '{figures_dir}'")

if __name__ == "__main__":
    main()
