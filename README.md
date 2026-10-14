# FinQA-RAG-LoRA
## Data

This project uses the FinQA subset of [T²-RAGBench](https://huggingface.co/datasets/G4KMU/t2-ragbench)
(CC BY 4.0): 8,281 questions over ~[N] pages from S&P 500 annual reports, each pairing text with a
financial table. Every question has a numeric answer and one gold page, so both retrieval and answer
accuracy can be scored automatically.

| Split | Questions | Use |
|---|---|---|
| train | 6,251 | LoRA training data (after distillation) |
| dev | 883 | Validation and prompt tuning |
| test | 1,147 | Final evaluation (fixed 300-question sample) |

**Key findings from [data exploration](notebooks/01_explore_data.ipynb):**

- **Clean split.** No page appears in both train and test, so fine-tuning can't memorize test pages.
- **Retrieval is hard within a company, not across companies.** 92% of test questions concern a company
  with 10+ pages (Entergy alone has 144). Filtering by company and year would cut ~2,800 candidates to a
  median of 2.
- **Answers need arithmetic.** Most questions require a difference, ratio, or percentage change rather
  than a lookup, which motivated having the model output a calculation that code evaluates.
- **No chunking needed.** Pages average ~1,100 tokens, all well under the 8,192-token embedding limit.
- **Labels are imperfect.** Cross-checking against the original FinQA answers and manually reviewing
  disagreements suggests at least 3–5% of labels are wrong (e.g. a sum reported instead of an average).
  Accuracy figures should be read against that ceiling.
- **Inconsistent formats.** Percentages are usually fractions (0.115 = 11.5%) but not always, so results
  report both strict and format-tolerant accuracy.