=====
PyBPR
=====

A Python implementation of Bayesian Personalized Ranking (BPR) for recommender systems with support for various negative sampling techniques and weighted Alternating Least Squares (WALS).

Overview
========

PyBPR is a comprehensive recommender systems library that implements:

* **Bayesian Personalized Ranking (BPR)** - A pairwise ranking approach for implicit feedback
* **Multiple Negative Sampling Strategies** - Various approaches for selecting negative examples
* **Viewed/Clicked BPR Variants** - Specialized implementations for different interaction types

The library is designed for researchers and practitioners working with implicit feedback data in recommender systems, where the goal is to learn personalized rankings of items for users.

Features
========

* **BPR Implementation**: Core Bayesian Personalized Ranking algorithm
* **Flexible Negative Sampling**: Multiple strategies for negative example selection
* **Interaction-Specific Models**: Variants optimized for viewed/clicked data
* **Research-Oriented**: Designed for experimentation and comparison of techniques

Installation
============

Prerequisites
-------------

PyBPR requires Python 3.6 or later and the following dependencies:

* NumPy
* SciPy
* Pandas (for data handling)
* Scikit-learn (for evaluation metrics)

Quick Install
-------------

1. Clone the repository::

    git clone https://github.com/rimplesandhu/pybpr.git
    cd pybpr

2. Install the package::

    pip install .

Development Install
-------------------

For development or if you want to modify the code::

    pip install -e .

This creates an editable installation that reflects changes to the source code immediately.

Virtual Environment (Recommended)
----------------------------------

It's recommended to install PyBPR in a virtual environment::

    # Create virtual environment
    python -m venv pybpr_env
    
    # Activate virtual environment
    # On Linux/Mac:
    source pybpr_env/bin/activate
    # On Windows:
    pybpr_env\Scripts\activate
    
    # Install PyBPR
    pip install .

Getting Started
===============

Basic BPR Usage
---------------

.. code-block:: python

    from pybpr import BPR
    import numpy as np
    
    # Load your user-item interaction matrix
    # interactions should be a sparse matrix (users x items)
    
    # Initialize BPR model
    model = BPR(
        factors=50,          # Number of latent factors
        learning_rate=0.01,  # Learning rate
        regularization=0.001, # Regularization parameter
        iterations=100       # Number of training iterations
    )
    
    # Train the model
    model.fit(interactions)
    
    # Get recommendations for a user
    user_id = 0
    recommendations = model.recommend(user_id, N=10)

Research Applications
=====================

This library is particularly useful for:

* **Comparing negative sampling strategies** in BPR
* **Studying interaction-specific models** (views vs. clicks)
* **Benchmarking different matrix factorization approaches**

Examples and Tutorials
======================

Check the ``examples/`` directory for:

* Basic BPR tutorial
* Negative sampling comparison
* WALS vs. BPR performance analysis
* Evaluation metrics walkthrough


Citation
========

If you use PyBPR in your research, please cite:

.. code-block:: bibtex

    @software{pybpr,
        title={PyBPR: Bayesian Personalized Ranking for Python},
        author={Rimple Sandhu and Charles Tripp},
        year={2024},
        url={https://github.com/rimplesandhu/pybpr}
    }

Contact
=======

* **Rimple Sandhu** - National Renewable Energy Laboratory - rimple.sandhu@nrel.com
* **Charles Tripp** - National Renewable Energy Laboratory - charles.tripp@nrel.gov

For questions, issues, or collaborations, please open an issue on GitHub or contact the maintainers directly.

Acknowledgments
===============

* Built on research from Rendle et al. (2009) - "BPR: Bayesian Personalized Ranking from Implicit Feedback"
* Inspired by the broader recommender systems research community
* Developed at the National Renewable Energy Laboratory