# Beta–Binomial classification

## Abstract

- **Beta–Binomial classification** is a **Bayesian model for binary outcomes** where the probability of success varies across groups and follows a Beta distribution.
- It’s mainly used when you have **count data with extra variability**.
- Next step:  a **mini 3-layer roadmap**:
  1. Phase 1 — use Beta–Binomial for classification (like your current setup)
  2. Phase 2 — integrate IRT or logistic regression for difficulty-aware modeling
  3. Phase 3 — wrap it with a contextual bandit to optimize learning resource delivery.
- Practical conclusion:
  - **Beta–Binomial classification** is an **excellent “first-stage mastery classifier”** — lightweight, Bayesian, interpretable, and easy to tune.
  - But for *modeling learning over time* or *content difficulty*, it should be **paired or extended** (e.g., with IRT, BKT, or bandit models).

---

## 1. The Big Picture

**Beta–Binomial classification** is a **Bayesian probabilistic model** used when you want to model **discrete outcomes (success/failure)** across **many groups or trials**, especially when:

* The number of successes out of trials follows a **Binomial distribution**, and
* The **success probability (p)** itself **varies across groups** and is modeled by a **Beta distribution**.

So it’s a **hierarchical Bayesian model** — a bridge between the **Binomial** and **Beta** distributions.

---

## 2. Core Intuition

Think of it like this:

> Each group (say, each student, patient, product, or user) has its own unknown probability of success, $p_i$.
> But instead of assuming $p_i$ is fixed, you assume it's drawn from a Beta distribution with parameters $\alpha$ and $\beta$.

Then:
```math
\begin{align*}
p_i &\sim \text{Beta}(\alpha, \beta) \\
y_i | p_i &\sim \text{Binomial}(n_i, p_i)
\end{align*}
```

Marginalizing out $p_i$ gives:
```math
y_i \sim \text{Beta-Binomial}(n_i, \alpha, \beta)
```

This allows **extra variability** (“overdispersion”) compared to a plain Binomial model.

---

## 3. Why It Matters

The **plain Binomial** assumes a fixed success probability ( p ). But in many real cases, success rates vary among individuals, items, or contexts.

Examples include marketing campaigns, student ability, or trial-site heterogeneity. **Beta–Binomial** acknowledges this **heterogeneity** — it’s a **Binomial with uncertainty about p**.

---

## 4. Typical Use Cases

- A/B testing with group variability
- Education/psychometrics for mastery from counts
- Reliability/manufacturing defect rates
- Sports win rates with limited data

---

## 5. Parameter Interpretation

- $\alpha$, $\beta$: Beta shape parameters
- Mean: $\mu = \frac{\alpha}{\alpha + \beta}$
- Variance: $\frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$
- Overdispersion grows as $\alpha + \beta$ decreases

---

## 6. Comparison to Logistic Regression

- Generative Bayesian vs. discriminative
- Naturally handles grouped counts and overdispersion
- Interpretable priors vs. weight-based interpretation

---

## 7. Strengths and 8. Weaknesses

Strengths: overdispersion handling, hierarchical structure, uncertainty, conjugacy, small-sample robustness.

Weaknesses: limited covariates without extensions, scalability, Beta prior assumptions, binary-only.

---

## 9. Extensions

- Beta–Binomial Regression
- Dirichlet–Multinomial for multiclass
- Hierarchical GLMs (PyMC/Stan)

---

## 10. Tooling / Implementation

```python
import pymc as pm

with pm.Model() as model:
    alpha = pm.HalfNormal('alpha', sigma=2)
    beta = pm.HalfNormal('beta', sigma=2)
    p = pm.Beta('p', alpha, beta, shape=n_groups)
    y = pm.Binomial('y', n=n_i, p=p, observed=y_i)
    trace = pm.sample()
```

---

## Educational Applications and Recommendations

- Excellent first-stage mastery classifier for small data
- Pair with BKT/IRT for time/difficulty modeling
- Roadmap: start with Beta–Binomial → add difficulty-awareness → add bandits for content selection
