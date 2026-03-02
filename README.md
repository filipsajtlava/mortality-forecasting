# Human Mortality Database (HMD) Modeling & Forecasting

This repository contains a Python-based pipeline for fetching, processing, and modeling demographic data from the [Human Mortality Database (HMD)](https://www.mortality.org/).

**Status:** Work in Progress

## Features

- **Automated Data Retrieval:**
  - Automatically handles web session authentication,
  - Downloads dataset files directly from HMD.

- **Stochastic Forecasting:**
  - Implements Lee-Carter with support for mutlidimensional monte carlo simulations.

## Technologies used
  - **Python** 3.8+
  - **pandas**, **numpy**, **xarray**
  - **matplotlib**

## Future development
  - Adding the Poisson GNM model,
  - Renshaw-Haberman, Cairns–Blake–Dowd, Li-Lee and other models,
  - Implementing LSTMs in place of the classic random walk with drift,
  - CNNs for cohort heatmap search.
