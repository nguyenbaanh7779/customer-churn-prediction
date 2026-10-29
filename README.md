# Customer Churn Prediction (Online Retail II)

Customer churn prediction and retention marketing targeting analysis using the UCI Online Retail II dataset.

## Overview

This project builds a system to predict whether an existing customer is likely to become inactive within a defined future period, so that a marketing team can prioritize retention spend on the customers most at risk of churning. Three targeting strategies are compared: a random/no-model baseline, a rule-based RFM approach, and a machine-learning model trained on customer purchase history.

The full write-up — business understanding, data preparation, modeling, and evaluation — lives in [`report/`](report/) as a LaTeX report (`report/main.tex`). A 2-page executive summary is in [`brief/`](brief/), and a slide deck is in [`slide/`](slide/).

## Repository Structure

```
data/raw/    Raw UCI Online Retail II transaction data (online_retail_II.xlsx)
notebooks/   Exploratory analysis and modeling notebooks
report/      LaTeX source for the full project report
brief/       LaTeX source for the 2-page executive brief
slide/       LaTeX (Beamer) source for the presentation deck
docs/guide/  Setup and usage guides
```

## Getting Started

See [docs/guide/README.md](docs/guide/README.md) for:
- setting up the Python environment used by the notebooks
- installing and compiling the LaTeX report

## Status

Early stage — data exploration is in progress; the report structure is in place and being filled in section by section.
