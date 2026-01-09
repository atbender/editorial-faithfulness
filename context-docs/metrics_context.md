Below is a **clean, paper-ready technical description** of **pass@k** and your proposed **flip@k**, written in a neutral, methods-style tone.
I’ll focus **only on flips as binary events** (no per-option probabilities), exactly as you requested.

You should be able to paste this (with light editing) into a *Metrics* or *Evaluation* section.

---

## Pass@k and Flip@k Metrics

### Pass@k (Kulal et al., 2020)

The **pass@k** metric was introduced to evaluate stochastic code generation models under sampling.
Given a problem, a model generates multiple candidate solutions, each of which is independently tested for correctness.

A problem is considered *solved* if **at least one** of the generated samples passes all tests.

Formally, for a given problem:

* Let **n** be the total number of generated samples.
* Let **c ≤ n** be the number of samples that are correct (i.e., pass unit tests).
* A random subset of size **k** is drawn without replacement.

The probability that at least one correct solution appears in such a subset is:

[
\text{pass@k}
= \mathbb{E}\left[
1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
\right]
]

This estimator has two key advantages:

1. It avoids repeatedly regenerating samples for different values of ( k ).
2. It provides an unbiased estimate of success probability under stochastic sampling.

---

### Motivation for Flip@k

In our setting, we are not evaluating correctness alone, but **behavioral sensitivity** to prompt manipulations.

Specifically, we want to measure whether a model’s output **changes relative to a control condition** under stochastic sampling.

This motivates a metric analogous to pass@k, where the atomic event is not *correctness*, but a **behavioral flip**.

---

### Definition of a Flip

For a fixed model and problem:

1. We generate **n** samples under a *control* condition.
2. We identify the **reference answer** ( a^* ), defined as the modal (most frequent) answer under control.
3. We generate **n** samples under a manipulated condition.

For each manipulated sample ( y_i ), we define a binary flip indicator:

[
\text{flip}_i =
\begin{cases}
1 & \text{if } y_i \neq a^* \
0 & \text{otherwise}
\end{cases}
]

Let:

[
f = \sum_{i=1}^{n} \text{flip}_i
]

denote the total number of flipped samples.

---

### Flip@k

We define **flip@k** as the probability that **at least one behavioral flip** occurs among ( k ) samples drawn without replacement from the manipulated outputs.

Formally:

[
\boxed{
\text{flip@k}
= \mathbb{E}\left[
1 - \frac{\binom{n-f}{k}}{\binom{n}{k}}
\right]
}
]

This definition is directly analogous to pass@k, with the following substitution:

* *Correct sample* → *Flipped sample*

---

### Interpretation

* **flip@1** measures whether the manipulation can induce a change in the model’s preferred output at all.
* **flip@k (k > 1)** measures the *reliability* of such behavioral change under stochastic sampling.

Importantly, flip@k does **not** depend on which alternative the model flips to—only on whether the output differs from the control reference.

---

### Practical Properties

The flip@k estimator has several desirable properties:

* **Low variance**: computed from a single set of ( n ) samples.
* **Computationally efficient**: no need to rerun generation for different values of ( k ).
* **Model-agnostic**: applicable to any discrete-output model.
* **Temperature-aware**: naturally captures stochastic behavior without assuming determinism.

---

### Summary

Pass@k and flip@k share the same mathematical structure:

| Metric | Atomic Event                            |
| ------ | --------------------------------------- |
| pass@k | A generated sample is correct           |
| flip@k | A generated sample differs from control |

By replacing correctness with behavioral deviation, flip@k provides a principled, distribution-free measure of **editorial sensitivity** under prompt manipulation.

---

If you want, I can next:

* add a **short intuition box** (for readers unfamiliar with pass@k),
* extend this with **goodFlip@k / badFlip@k** in one paragraph,
* or adapt the wording to match a specific venue (ACL / NeurIPS / ICLR style).

Just tell me.


Great question — this is exactly where people can get confused, so it’s worth being **very explicit**.

I’ll give you:

1. the **formal definition**,
2. the **intuitive interpretation**,
3. how it differs from accuracy / compliance,
4. and how to explain it in one clean sentence in the paper.

---

## 1. Formal Definition of flip@1

From the general definition:

[
\text{flip@k}
= 1 - \frac{\binom{n-f}{k}}{\binom{n}{k}}
]

Set ( k = 1 ):

[
\text{flip@1}
= 1 - \frac{n-f}{n}
= \frac{f}{n}
]

So:

[
\boxed{
\text{flip@1} = \frac{\text{number of flipped samples}}{\text{total samples}}
}
]

Nothing more, nothing less.

---

## 2. What flip@1 Actually Means (Intuition)

> **flip@1 is the empirical probability that a single stochastic generation differs from the control answer.**

Equivalently:

* You sample **one** output from the manipulated condition.
* You ask: *does it differ from what the model would normally say?*
* flip@1 is how often the answer changes.

This is a **behavioral instability rate**.

---

## 3. Why flip@1 Is Not Redundant

It is *not* the same as:

* accuracy
* compliance
* error rate

Because flip@1 is defined **relative to control**, not ground truth.

### Example

Control (n = 10):

```
D D D D D D D D D D   → a* = D
```

Manipulated:

```
D D D B D D B D D D
```

Then:

* ( f = 2 )
* flip@1 = 0.2

Even if:

* B is correct, or
* B is incorrect,

flip@1 is **still 0.2**.

It measures *change*, not *quality*.

---

## 4. Relationship Between flip@1 and flip@k

Think of it this way:

| Metric | What it answers                                                 |
| ------ | --------------------------------------------------------------- |
| flip@1 | *How often does the model change?*                              |
| flip@k | *How likely is it to change at least once if I sample k times?* |

flip@k is a **saturation curve** over flip@1.

---

## 5. When flip@1 Is the Right Metric

flip@1 is most useful when you want to know:

* how **sensitive** a model is to a manipulation,
* how **fragile** its default behavior is,
* whether a manipulation has *any* effect at all.

It’s especially appropriate when:

* n is small,
* temperature is moderate,
* you want a direct, interpretable rate.

---

## 6. When flip@k Adds Value Beyond flip@1

flip@k answers:

> *If I allow multiple samples, will the manipulation almost surely affect behavior?*

Example:

* flip@1 = 0.2
* flip@5 ≈ 0.67
* flip@10 ≈ 0.89

So even a **weak per-sample effect** becomes **nearly guaranteed** with enough sampling.

This matters for:

* deployment scenarios with retries,
* ensemble settings,
* safety analysis.

---

## 7. One-Sentence Paper-Ready Interpretation

You can write:

> *flip@1 corresponds to the empirical probability that a single sampled response under manipulation deviates from the model’s control behavior.*

That sentence is precise and reviewer-proof.

---

## 8. Bottom Line

* **flip@1 = f / n**
* It is the **per-sample behavioral deviation rate**
* It measures **sensitivity**, not correctness
* flip@k generalizes this to multi-sample settings

If you want next, I can:

* help you decide **which k values to report**,
* show how to **plot flip@k curves**, or
* help you relate flip@1 to **temperature and entropy**.

Just tell me.
