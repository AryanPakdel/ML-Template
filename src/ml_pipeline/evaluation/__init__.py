"""Evaluation stage: metrics, diagnostic plots, and error analysis.

Modules are imported directly by their users (trainer, leaderboard, reports);
this package intentionally re-exports nothing so optional heavy dependencies
(matplotlib, shap) are only paid for when actually used.
"""
