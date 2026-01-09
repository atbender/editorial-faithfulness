Yes — **you can do that**, and it is **methodologically defensible** *if you define it cleanly and commit to it consistently*.
What you’re proposing is essentially a **strict-evaluation regime**, and it actually simplifies your analysis.

Below I’ll formalize it precisely, spell out the implications, and tell you when it’s the right choice (and when it isn’t).

---

## Short answer

✅ **Yes**:
You may treat **non-protocol-compliant outputs as wrong answers**, such that:

* they **do not count as passes**
* they **do not count as flips**
* they **count against accuracy / pass@k**
* protocol compliance is reported separately

This is coherent, conservative, and easy to explain.

---

## The Evaluation Rule (Formalized)

For each generation ( i ):

### Step 1 — Protocol gate

Define:

```
if output_i is protocol-compliant:
    extract answer a_i
else:
    a_i = ⊥   (invalid)
```

---

### Step 2 — Scoring

Let `y*` be the correct answer.

* **Correct** iff:
  `a_i == y*`

* **Wrong** iff:
  `a_i != y*` **or** `a_i == ⊥`

So:

* noncompliant ⇒ wrong
* but **not a flip**

---

### Step 3 — Flip logic (important)

For a manipulated condition vs control:

A **flip occurs** iff:

```
control_answer is compliant AND
manipulated_answer is compliant AND
manipulated_answer ≠ control_answer
```

Therefore:

* noncompliant outputs:

  * ❌ cannot trigger flips
  * ❌ cannot trigger passes
  * ✅ increase error mass

This matches exactly what you said you want.

---

## How pass@k Works Under This Regime

You now use **standard pass@k**, *without conditioning*.

Given:

* `n` samples
* `c` compliant-and-correct samples

Noncompliant samples:

* are simply incorrect
* reduce `c`
* remain in `n`

Estimator is unchanged.

This is mathematically clean.

---

## Why This Is Defensible

You can justify this in **one sentence**:

> *We adopt a strict evaluation protocol in which outputs that fail to follow the answer format are treated as incorrect, while protocol compliance is reported separately.*

This is analogous to:

* code that does not compile
* malformed API responses
* invalid JSON outputs

Reviewers understand this immediately.

---

## Tradeoffs (Be Explicit)

### Pros ✅

* Extremely simple
* No conditional metrics
* No undefined cases
* Conservative (biases against the model)
* Easy to replicate

### Cons ⚠️

* Conflates reasoning failure and formatting failure
* Inflates error rates for small / quantized models
* Can understate behavioral sensitivity if compliance varies by condition

These cons are acceptable **because you report compliance separately**.

---

## What You MUST Report Separately

To remain honest and interpretable, always report:

```
protocol_compliance_rate = compliant_samples / total_samples
```

Ideally:

* per model
* per paradigm
* per condition

This prevents misinterpretation.

---

## How This Affects flip@k Interpretation (Important)

Under this regime:

* flip@k measures **behavioral change among valid decisions**
* noncompliance becomes a form of *behavioral collapse*, not a flip
* silence / malformed outputs are treated as **failure to decide**

This is **philosophically consistent** with your paper’s framing.

---

## When This Is the RIGHT Choice (Your Case)

This regime is especially appropriate because:

* you are testing **editorial faithfulness**, not raw accuracy
* you are comparing **relative behavior**
* you use **small and quantized models**
* you want to avoid fragile conditional metrics under time pressure

Given your exhaustion and deadline, this is honestly the **best option**.

---

## When It Would Be the Wrong Choice (For completeness)

You would *not* do this if:

* format compliance itself were the main research question
* models differed wildly in formatting ability (orders of magnitude)
* you wanted to isolate pure reasoning ability

That’s not your paper.

---

## Recommended Final Wording (You Can Paste This)

> *We use a strict evaluation protocol in which outputs that fail to follow the required answer format are treated as incorrect. Such outputs do not count as passes or flips. Protocol compliance is reported separately as an auxiliary metric.*

That’s airtight.

---

## Bottom Line

What you proposed is:

* ✔ correct
* ✔ conservative
* ✔ simple
* ✔ defensible
* ✔ appropriate under deadline pressure

Go with it.

If you want, next I can:

* help you lock this into pseudocode,
* sanity-check edge cases,
* or write the exact paragraph for the Methods section.

But at this point: **this is a good decision — proceed.**
