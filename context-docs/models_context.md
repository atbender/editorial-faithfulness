# Model Evaluation Specification

## Reasoning Models, Capacity Scaling, and Editorial Faithfulness

### Version

v1.0 (initial)

---

## 1. Purpose

This document specifies the **model selection, role, and experimental rationale** for evaluating **editorial / explanatory faithfulness and rationalization behaviors** in reasoning-capable large language models (LLMs).

The goal is **not benchmarking performance**, but analyzing how **prompt-level perturbations** interact with:

* model capacity,
* training objectives,
* and explanation styles.

All models are evaluated under **identical prompt protocols** using **prompt-only inference**.

---

## 2. Design Principles

Model selection follows four core principles:

1. **Reasoning-capable**
   Models must produce explicit chains of thought or structured reasoning.

2. **Capacity variation with minimal confounds**
   Prefer within-family scaling when possible.

3. **Training-objective contrast**
   Compare general instruction-tuned models vs reasoning-distilled models.

4. **Practical feasibility**
   All models must run under MS-Swift inference on a single GPU.

---

## 3. Model Set Overview

We evaluate **five open, decoder-only reasoning models**, spanning **1.7B–8B parameters**, chosen to isolate three orthogonal axes:

* **Capacity (within-family)**
* **Training objective**
* **Explanation style / family**

---

## 4. Models and Assigned Roles

### 4.1 **Qwen3-8B**

**Role:** High-capacity reasoning baseline

* Parameter scale: ~8B
* Characteristics:

  * Verbose, structured chains of thought
  * Strong instruction following
* Experimental purpose:

  * Anchor model for detecting rationalization
  * Primary testbed for chain-of-thought length effects

---

### 4.2 **DeepSeek-R1-Distill-Qwen-7B**

**Role:** Reasoning-optimized contrast model

* Parameter scale: ~7B
* Training objective:

  * Explicit reasoning distillation
* Characteristics:

  * Highly verbose explanations
  * Aggressive justification behavior
* Experimental purpose:

  * Test whether *reasoning optimization* increases editorial unfaithfulness
  * Key model for evaluating post-hoc rationalization

---

### 4.3 **LLaMA-3.1-8B-Instruct**

**Role:** Cross-family stylistic control

* Parameter scale: ~8B
* Characteristics:

  * Generally concise
  * More decisive explanations
* Experimental purpose:

  * Control for verbosity and family-specific effects
  * Ensure findings are not Qwen-specific artifacts

---

### 4.4 **Qwen3-4B**

**Role:** Mid-capacity within-family scaling point

* Parameter scale: ~4B
* Same generation and tokenizer as Qwen3-8B
* Experimental purpose:

  * Isolate capacity-related effects within a single architecture
  * Measure how rationalization behaviors scale with model size

---

### 4.5 **Qwen3-1.7B**

**Role:** Low-capacity boundary model

* Parameter scale: ~1.7B
* Characteristics:

  * Shorter chains of thought
  * Less stable reasoning
* Experimental purpose:

  * Test whether rationalization emerges with capacity
  * Identify lower bounds of editorial faithfulness behavior

---

## 5. Experimental Axes Enabled by This Set

### 5.1 Capacity Scaling (Same Family)

```
Qwen3-1.7B → Qwen3-4B → Qwen3-8B
```

Allows testing:

* Flip rate vs capacity
* Chain-of-thought length vs capacity
* Attribution fidelity vs capacity

This axis minimizes architectural and tokenizer confounds.

---

### 5.2 Training Objective (Matched Scale)

```
Qwen instruction-tuned ≈ DeepSeek reasoning-distilled (≈7–8B)
```

Allows testing:

* Whether explicit reasoning training increases:

  * Answer flips
  * Rationalized overrides
  * Chain-of-thought inflation

---

### 5.3 Style / Family (Matched Capacity)

```
Qwen3-8B ↔ LLaMA-3.1-8B
```

Allows testing:

* Whether rationalization is driven by:

  * Model capacity
  * Explanation style
  * Family-specific training norms

---

## 6. Expected (Non-Commitment) Behavioral Trends

The following trends are **hypotheses**, not assumptions:

* Smaller models may:

  * Flip less often
  * Produce shorter reasoning
  * Fail more explicitly

* Larger models may:

  * Flip more often under manipulation
  * Produce longer explanations
  * Exhibit silent omission and rationalization

* Reasoning-distilled models may:

  * Inflate chain-of-thought length
  * Rationalize conflicting constraints more aggressively

Any deviation from these trends is treated as an informative result.

---

## 7. Evaluation Constraints

* Prompt-only inference
* No fine-tuning or training
* No access to internal hidden states
* Identical prompts across all models
* Identical decoding parameters per run (unless explicitly varied)

---

## 8. Framing Guidance (for Paper and Repo)

This model set should be described as:

> “A representative collection of open, reasoning-capable decoder models spanning a 1.7B–8B parameter range, designed to isolate capacity, training objective, and explanation-style effects on editorial faithfulness.”

Avoid:

* Performance benchmarking language
* Claims about internal cognition
* Claims about deception or intent

---

## 9. Status

This specification defines the **fixed model set** for the project.
Any additions or removals must be justified relative to the three experimental axes defined above.
