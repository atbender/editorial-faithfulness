# Technical Context Document

## Psychology-Inspired Prompt-Based Evaluation of Faithfulness and Rationalization in Reasoning LLMs

### 1. Purpose & High-Level Goal

The goal of this project is to **evaluate editorial faithfulness and rationalization behaviors in reasoning-capable large language models (LLMs)** using **prompt-only, psychology-inspired experimental protocols**.

We aim to determine whether:

1. Psychological experimental paradigms can be meaningfully adapted to decoder-only LLMs.
2. Behavioral changes induced by prompt manipulations are *faithfully attributed* in the model’s reasoning.
3. **Post-hoc rationalization** can be detected via **chain-of-thought (CoT) properties**, particularly length.

This work is designed to be:

* Prompt-only (no training or fine-tuning)
* Model-agnostic
* Based on **synthetic stimulus sets**, not natural datasets
* Executable within a 48-hour experimental window

---

### 2. Core Research Questions

1. **Faithfulness**
   Do LLMs explicitly acknowledge and integrate prompt-level manipulations (tips, anchors, framing, etc.) into their reasoning?

2. **Rationalization**
   When LLM behavior changes, do models:

   * Attribute the change to the manipulation?
   * Or silently omit it while generating coherent post-hoc justifications?

3. **Chain-of-Thought Diagnostics**
   Is **increased CoT length** systematically associated with:

   * Answer flips
   * Unacknowledged manipulations
   * Rationalized overrides?

---

### 3. Central Hypotheses

#### H1 — Editorial Unfaithfulness

LLMs frequently exhibit **behavioral sensitivity** to prompt manipulations while **failing to explicitly attribute** their reasoning to those manipulations.

#### H2 — Rationalization Length Hypothesis

Answer flips induced by prompt manipulations are **associated with longer chains of thought**, especially when the manipulation is not explicitly acknowledged, suggesting post-hoc rationalization rather than direct constraint incorporation.

---

### 4. Experimental Paradigms (Psychology-Inspired)

All experiments follow a shared structure:

* Controlled manipulation
* Behavioral measurement
* Attribution and reasoning analysis

#### 4.1 Tip / Constraint Injection (Baseline Paradigm)

* A relevant but non-trivial “tip” is added at the beginning of the prompt.
* Compare control vs tipped prompts.
* Measure:

  * Answer flip rate
  * Tip acknowledgment
  * CoT length

#### 4.2 Anchoring Effect

* Introduce an explicitly irrelevant numerical anchor.
* Measure:

  * Numeric drift toward anchor
  * Whether anchor is rejected, ignored, or implicitly used
  * CoT length differences

#### 4.3 Authority Bias

* Present a claim attributed to a “leading expert” suggesting a suboptimal approach.
* Measure:

  * Adoption of authority’s suggestion
  * Attribution to authority
  * Rationalization patterns

#### 4.4 Choice Blindness

* Step 1: Model chooses between A/B.
* Step 2: Model is told it chose the opposite option.
* Measure:

  * Acceptance vs correction
  * Rationalization behavior
  * CoT inflation in acceptance cases

*(Only a subset—typically 2–3 paradigms—may be used depending on time.)*

---

### 5. Synthetic Stimulus Sets (Not Datasets)

* Experiments use **small, controlled synthetic stimulus sets**, analogous to stimuli in experimental psychology.
* Typical scale:

  * 5–10 stimuli per paradigm
  * 2–3 paradigms
  * 2–4 models
* Goal is **low semantic entropy**, not task realism.

Stimuli are designed to:

* Be plausible
* Avoid trivial correctness
* Enable controlled perturbation analysis

---

### 6. Unified Metrics (Logged Per Response)

Each response should log:

| Metric                 | Description                       |
| ---------------------- | --------------------------------- |
| Model                  | Model identifier                  |
| Paradigm               | Tip / Anchor / Authority / Choice |
| Condition              | Control / Manipulated             |
| Behavioral Outcome     | Answer, estimate, or choice       |
| Answer Flip            | Boolean (vs control)              |
| Manipulation Mentioned | Explicit / Implicit / No          |
| Manipulation Used      | Yes / No                          |
| CoT Length             | Token count (preferred)           |
| Rationalization Flag   | Manual or heuristic               |
| Notes                  | Optional qualitative annotation   |

---

### 7. Outcome Grouping (Critical for Analysis)

For manipulated prompts:

* **G₀**: No flip vs control
* **G₁**: Flip with explicit acknowledgment
* **G₂**: Flip without acknowledgment (rationalized override)

Primary comparisons:

* CoT length: flipped vs non-flipped
* CoT length: G₂ > G₁
* Manipulated vs control (sanity check)

---

### 8. Analysis Strategy

* Report:

  * Mean, median, variance of CoT length
  * Flip and attribution rates
* Use lightweight statistics:

  * Non-parametric tests or bootstrap CIs
* Include qualitative examples illustrating:

  * Silent omission
  * Rationalized override
  * Faithful incorporation

---

### 9. Key Constraints

* No model training or fine-tuning
* No reliance on internal hidden states
* No large datasets
* Prompt-only, black-box access
* Results focus on **relative behavior**, not correctness

---

### 10. Intended Contribution

1. A **prompt-based evaluation protocol** for editorial faithfulness in LLM reasoning.
2. Evidence that **classic psychological paradigms** can be adapted to LLMs.
3. Empirical support for **chain-of-thought length as a diagnostic signal** of rationalization.

---

### 11. Intended Use of This Document

This document is intended to:

* Provide **context and goals** to an automated coding / experimentation agent
* Guide stimulus generation, experiment execution, and logging
* Ensure alignment with the paper’s conceptual and methodological goals