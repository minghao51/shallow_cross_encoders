# 1-bit Mamba for Query Rewriting (Exploratory)

> **Status:** Speculative research note. Not implemented in the current codebase. Explores using 1-bit Mamba models for query normalization/expansion to improve static embedding retrieval.

## 1. Why 1-bit Mamba is the Right Tool
Standard Transformers (like GPT-4o-mini) are overkill and too slow for a real-time query rewrite on a CPU.
* **Linear Scaling:** Mamba-3 architectures have linear complexity. Unlike Transformers, they don't get bogged down as the query gets longer.
* **1-Bit Efficiency:** A binarized model (Weights $\in \{-1, 1\}$ or ternary $\{-1, 0, 1\}$) performs almost no floating-point multiplications. It's mostly addition and bit-shifting, which CPUs handle with extreme efficiency.
* **The Result:** You can run a "Normalization" pass in **<15ms** on a standard server CPU thread.

## 2. The Two Deployment Patterns

### Pattern A: The "Normalizer" (Fixing the Potion Gap)
Since Potion-style embeddings are extremely sensitive to typos, you use the 1-bit model to "translate" the query into a clean version.
* **User Query:** *"Who is the auther of the paper on 1bit mamba?"*
* **Mamba Correction:** *"Who is the author of the paper on 1-bit Mamba?"*
* **Embedding Step:** You embed the *corrected* query. This ensures the static vector for "auther" (which might be junk) is replaced by the high-quality vector for "author."

### Pattern B: The "Expander" (The RAG Power Play)
Instead of just fixing typos, you have the small model generate **synonyms** or **related entities** to help the "Basic" static model find better matches.
* **User Query:** *"HDB appreciation near top schools"*
* **Mamba Expansion:** *"Singapore public housing, property value growth, proximity to elite primary schools, Raffles Institution, Nan Hua."*
* **Utility:** This feeds your static model a "bag of keywords" that are much more likely to hit your vector index than the user's short, vague sentence.

---

## 3. The "Typo Rescue" Architecture
If you implement this, your retrieval pipeline looks like this:

| Stage | Component | Latency | Benefit |
| :--- | :--- | :--- | :--- |
| **Input** | Raw User Text | 0ms | High noise / potential typos. |
| **Correction** | **Bi-Mamba-0.7B** | ~12ms | Replaces typos with canonical terms. |
| **Retrieval** | Potion-32M (Static) | ~2ms | High recall on cleaned tokens. |
| **Rerank** | KineticRank (CPU) | ~80ms | Re-orders based on domain logic. |



---

## 4. Immediate Gaps & Implementation Advice
If you are moving forward with this, keep these "2026 hurdles" in mind:

1.  **Vocabulary Alignment:** Ensure your 1-bit model and your Potion embedding use the same tokenizer (or at least handle subwords similarly). If the 1-bit model corrects a word to something your static model doesn't have in its vocab, you're back to square one.
2.  **Over-Correction:** Small models can be "aggressive." For your healthcare projects, ensure the 1-bit model doesn't "correct" a rare medical acronym into a common word (e.g., correcting a specific drug name into a generic noun).
    * *Fix:* Use a "constrained" decoding or a very low temperature (0.1) for the correction task.
