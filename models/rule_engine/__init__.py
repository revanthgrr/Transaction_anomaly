# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Dynamic Rule Engine — Package Init
#
# Exposes the top-level `RuleEngine` orchestrator for easy imports.

# %%
from .engine import RuleEngine  # noqa: F401

__all__ = ["RuleEngine"]
__version__ = "1.0.0"
