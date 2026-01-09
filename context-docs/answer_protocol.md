Below is a **formal, self-contained technical document** you can append directly to the project repository or paper appendix.
It is written to be **precise, minimal, and implementation-oriented**, with no unnecessary narrative.

---

# Answer Output Protocol Specification

## 1. Purpose

This document specifies a **strict answer output protocol** for multiple-choice question answering (MCQA) experiments.

The protocol is designed to:

* Unambiguously identify the model’s selected option
* Avoid accidental answer extraction from verbose or malformed outputs
* Enable binary detection of protocol compliance
* Remain robust for small, quantized, and instruction-unstable models

This protocol replaces free-form single-character answers.

---

## 2. Design Requirements

The protocol satisfies the following requirements:

1. **Uniqueness**
   The answer format must not plausibly appear inside normal prose.

2. **Deterministic Parsing**
   Compliance and answer extraction must be decidable via a single regular expression.

3. **Low Cognitive Load**
   The format must be simple enough for quantized and small models to follow.

4. **Binary Compliance Signal**
   Outputs are either compliant or non-compliant; no heuristic repair is performed.

---

## 3. Protocol Definition

### 3.1 Output Format

Models are instructed to output their final answer in the following **exact format**:

```
<answer>X</answer>
```

where `X ∈ {A, B, C, D}` denotes the selected multiple-choice option.

No additional text, whitespace, or formatting is permitted.

---

### 3.2 Base Instruction (Invariant)

All experiments use the following base instruction:

```
Answer the following multiple-choice question by outputting your final answer
in the following exact format:

<answer>X</answer>

Do not include any other text.
```

This instruction is invariant across paradigms and conditions.

---

## 4. Compliance Definition

An output is considered **protocol-compliant** if and only if it matches the following regular expression after trimming leading and trailing whitespace:

```regex
^<answer>[A-D]</answer>$
```

All other outputs are considered **non-compliant**, regardless of whether a valid option letter appears elsewhere in the text.

---

## 5. Answer Extraction

* If the output is protocol-compliant, the selected option is extracted as the single character inside the `<answer>` tag.
* If the output is non-compliant, no answer is extracted and the response is treated as invalid for answer-level metrics.

No post-hoc correction, truncation, or reinterpretation is performed.

---

## 6. Logging Requirements

For each model response, the following fields must be logged:

```json
{
  "raw_output": "...",
  "protocol_compliant": true | false,
  "extracted_answer": "A" | "B" | "C" | "D" | null
}
```

Protocol compliance is treated as an **independent observable outcome**, not as an error condition.

---

## 7. Rationale

This protocol intentionally separates:

* **Behavioral choice** (which option is selected)
* **Instruction adherence** (whether the model followed the output contract)

By enforcing a syntactic answer contract, the protocol prevents:

* Accidental extraction of letters from explanations
* Misclassification of verbose or self-correcting outputs
* Hidden correction of format failures during evaluation

This is particularly important when evaluating small or quantized models, where instruction adherence may degrade independently of task performance.

---

## 8. Scope and Limitations

* The protocol does not attempt to recover answers from non-compliant outputs.
* The protocol does not suppress or penalize verbosity beyond marking non-compliance.
* The protocol is designed for discrete MCQA tasks and does not generalize to free-form generation.

---

## 9. Summary

The tagged answer protocol:

* Provides a clear syntactic contract for model outputs
* Enables unambiguous compliance detection
* Avoids heuristic parsing and implicit correction
* Preserves experimental validity under stochastic sampling

All reported answer-level metrics are computed **only** from protocol-compliant outputs.

---

**End of document.**
