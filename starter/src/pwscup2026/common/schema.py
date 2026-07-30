"""配布データの列定義。"""
from __future__ import annotations

# 共変量（9列）。
COVARIATES = ["age", "sex", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "smoking"]

CONTINUOUS = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]

# sex: 1=男, 0=女 ／ smoking: 現在喫煙 0/1
