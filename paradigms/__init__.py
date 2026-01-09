# Paradigm implementations for editorial faithfulness experiments

from .base import (
    Paradigm,
    ParadigmConfig,
    MCQAProblem,
    ExperimentalCondition,
    TrialResult,
    BASE_INSTRUCTION,
)

from .ethical_information_access import EthicalInformationAccessParadigm
from .authority_bias import AuthorityBiasParadigm
from .reframing_bias import ReframingBiasParadigm

__all__ = [
    # Base classes
    "Paradigm",
    "ParadigmConfig",
    "MCQAProblem",
    "ExperimentalCondition",
    "TrialResult",
    "BASE_INSTRUCTION",
    # Paradigms
    "EthicalInformationAccessParadigm",
    "AuthorityBiasParadigm",
    "ReframingBiasParadigm",
]

