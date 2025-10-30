<div align="center">
<h1>IRT Notebooks Collection</h1>
</div>

## Overview

This directory contains comprehensive Jupyter notebooks that demonstrate Item Response Theory (IRT) concepts through interactive examples, simulations, and analysis.

## Available Notebooks

### 1. IRT Data Generator (`1_IRT_Data_Generator.ipynb`)

**Purpose**: Generate synthetic IRT data for educational and research purposes.

**What it covers**:
- Understanding IRT models (1PL, 2PL, 3PL)
- Data generation process with realistic parameters
- Response simulation based on IRT probabilities
- Data export and visualization
- Model comparison and analysis

**Key Features**:
- Configurable model parameters
- Multiple IRT model support (1PL, 2PL, 3PL)
- Comprehensive data visualization
- Export capabilities for further analysis

**When to use**: Start here to generate synthetic IRT datasets

---

### 2. IRT Exploratory Data Analysis (`2_IRT_Exploratory_Analysis.ipynb`)

**Purpose**: Comprehensive exploratory analysis to understand and validate IRT data.

**What it covers**:
- Data loading and inspection
- Data quality assessment (missing values, outliers, integrity checks)
- Comprehensive statistical analysis
- Distribution analysis and visualizations
- Model comparison across 1PL, 2PL, and 3PL
- Correlation analysis between parameters

**Key Features**:
- Automated data quality checks
- Statistical summaries and descriptive statistics
- 9-panel comprehensive visualization dashboard
- Model comparison analysis
- Correlation heatmaps

**When to use**: After generating data, use this to verify data quality and gain insights

---

### 3. IRT Application and Analysis (`3_IRT_Application_Analysis.ipynb`)

**Purpose**: Advanced IRT concepts and practical applications through simulation.

**What it covers**:
- IRT probability and information functions
- Item Characteristic Curves (ICCs)
- Information function analysis
- Adaptive testing simulation
- Multi-student performance evaluation
- Convergence analysis

**Key Features**:
- Interactive visualizations
- Adaptive testing algorithms
- Performance metrics and evaluation
- Real-world application examples
- Ability estimation methods

**When to use**: After data exploration, use this for advanced analysis and applications

## Usage Instructions

**Recommended Workflow:**

1. **Generate Data** → Run `1_IRT_Data_Generator.ipynb` to create synthetic IRT datasets
2. **Explore & Validate** → Run `2_IRT_Exploratory_Analysis.ipynb` to verify data quality and gain insights  
3. **Apply & Analyze** → Run `3_IRT_Application_Analysis.ipynb` for advanced IRT applications
4. **Customize** → Modify parameters in any notebook to explore different scenarios
5. **Export** → Save generated data and visualizations for further research

## Requirements

- Python 3.7+
- Jupyter Notebook
- Required packages: numpy, pandas, matplotlib, seaborn, scipy

## Learning Path

1. **Beginner**: Start with `1_IRT_Data_Generator.ipynb` to understand basic IRT concepts and data generation
2. **Intermediate**: Use `2_IRT_Exploratory_Analysis.ipynb` to learn data analysis and quality assessment
3. **Advanced**: Explore `3_IRT_Application_Analysis.ipynb` for adaptive testing and advanced applications
4. **Expert**: Customize parameters across all notebooks to explore your own research questions

## Output Files

The notebooks generate various output files:
- **Data files**: CSV files with response matrices and parameters
- **Visualizations**: PNG files with charts and graphs
- **Results**: Summary statistics and analysis results

All output files are saved in the `../data/` and `../figures/` directories.