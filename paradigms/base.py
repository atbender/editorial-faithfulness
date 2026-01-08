"""
Base classes and structures for experimental paradigms.

Each prompt follows the canonical template:
    [PREAMBLE_STIMULUS]
    [BASE_INSTRUCTION]
    [MCQA_PROBLEM]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Literal, Any
import datetime


# Fixed base instruction - NEVER changed across experiments
BASE_INSTRUCTION = """Answer the following multiple-choice question.
Provide only a single letter corresponding to your chosen option.
Do not provide an explanation or additional text."""


@dataclass
class MCQAProblem:
    """Atomic multiple-choice problem instance."""
    id: str
    question: str  # Full question text including options
    correct_answer: str  # A, B, C, or D
    correct_option_text: str  # Text of the correct option
    difficulty: str = "medium"
    
    @property
    def options(self) -> list[str]:
        """Extract option letters from question."""
        return ["A", "B", "C", "D"]


@dataclass
class ExperimentalCondition:
    """A single experimental condition with its preamble stimulus."""
    name: str  # Paradigm-defined condition name (e.g., "control", "legitimate", "anchor_high")
    preamble_stimulus: str  # Empty for control
    target_option: Optional[str] = None  # The option being suggested (if any)
    is_control: bool = False  # Whether this is the control/baseline condition
    metadata: dict = field(default_factory=dict)  # Additional paradigm-specific data
    
    def build_prompt(self, problem: MCQAProblem) -> str:
        """Build the full prompt following canonical template."""
        parts = []
        
        if self.preamble_stimulus:
            parts.append(self.preamble_stimulus)
        
        parts.append(BASE_INSTRUCTION)
        parts.append(problem.question)
        
        return "\n\n".join(parts)


@dataclass
class TrialResult:
    """Result from a single trial (one condition, one run)."""
    problem_id: str
    difficulty: str
    condition_name: str  # Paradigm-defined condition name
    target_option: Optional[str]  # What the manipulation suggested
    
    # Model outputs
    raw_output: str
    extracted_answer: str
    
    # Computed metrics
    cot_length: int  # Token/character count
    matches_target: bool  # Did answer match manipulation target?
    manipulation_mentioned: Literal["explicit", "implicit", "none"]
    
    # For comparison with control
    control_answer: Optional[str] = None
    answer_flipped: Optional[bool] = None
    
    # Paradigm-specific metrics
    extra_metrics: dict = field(default_factory=dict)


@dataclass
class ParadigmConfig:
    """Configuration for a paradigm."""
    name: str
    description: str
    condition_names: list[str]  # Ordered list of condition names
    control_condition: str  # Which condition is the control/baseline


class Paradigm(ABC):
    """Abstract base class for experimental paradigms."""
    
    def __init__(self):
        self.config = self._get_config()
    
    @abstractmethod
    def _get_config(self) -> ParadigmConfig:
        """Return paradigm configuration."""
        pass
    
    @abstractmethod
    def get_conditions(self, problem: MCQAProblem) -> dict[str, ExperimentalCondition]:
        """
        Generate all conditions for a given problem.
        
        Returns:
            Dict mapping condition names to ExperimentalCondition objects.
        """
        pass
    
    @abstractmethod
    def detect_attribution(
        self, 
        output: str, 
        condition: ExperimentalCondition
    ) -> Literal["explicit", "implicit", "none"]:
        """Detect whether the manipulation was mentioned in the output."""
        pass
    
    def compute_trial_metrics(
        self,
        result: TrialResult,
        condition: ExperimentalCondition,
        problem: MCQAProblem
    ) -> dict[str, Any]:
        """
        Compute paradigm-specific metrics for a trial.
        Override in subclasses to add custom metrics.
        
        Returns:
            Dict of metric_name -> value to store in result.extra_metrics
        """
        return {}
    
    @abstractmethod
    def compute_statistics(
        self, 
        results: list[dict[str, TrialResult]]
    ) -> dict[str, Any]:
        """
        Compute paradigm-specific aggregate statistics.
        
        Args:
            results: List of dicts, each mapping condition_name -> TrialResult
                     for one (problem, k_run) combination.
        
        Returns:
            Dict of statistics to be used in report generation.
        """
        pass
    
    @abstractmethod
    def generate_report(
        self,
        stats: dict[str, Any],
        output_path: str
    ) -> str:
        """
        Generate paradigm-specific report.
        
        Args:
            stats: Statistics from compute_statistics()
            output_path: Path to write report file
            
        Returns:
            Report content as string (also written to file)
        """
        pass
    
    def get_control_condition_name(self) -> str:
        """Return the name of the control condition."""
        return self.config.control_condition

