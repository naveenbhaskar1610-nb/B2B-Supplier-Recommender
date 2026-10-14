# Evaluation results

Synthetic catalogue, 36 hand-labelled queries. See `evaluate.py` for caveats.

## 1. Retrieval ablation (precision@5 and hit@1)

| variant                       | metric   |   plain |   typo |   jargon |   unseen |   overall |
|:------------------------------|:---------|--------:|-------:|---------:|---------:|----------:|
| A word TF-IDF only            | p@5      |       1 |   0.75 |      0.3 |     0.62 |      0.67 |
| A word TF-IDF only            | hit@1    |       1 |   0.75 |      0.3 |     0.62 |      0.67 |
| B + synonyms                  | p@5      |       1 |   0.75 |      1   |     0.62 |      0.86 |
| B + synonyms                  | hit@1    |       1 |   0.75 |      1   |     0.62 |      0.86 |
| C + char n-grams              | p@5      |       1 |   1    |      0.4 |     0.88 |      0.81 |
| C + char n-grams              | hit@1    |       1 |   1    |      0.4 |     0.88 |      0.81 |
| D hybrid + synonyms (default) | p@5      |       1 |   1    |      1   |     0.88 |      0.97 |
| D hybrid + synonyms (default) | hit@1    |       1 |   1    |      1   |     0.88 |      0.97 |

## 2. Ranking layer vs raw relevance order (mean over queries, top-5 suppliers)

| method                |   precision@5 |   gst_verified_share |   avg_response_rate |   avg_reply_hours |   avg_rating |
|:----------------------|--------------:|---------------------:|--------------------:|------------------:|-------------:|
| multi-criteria ranker |         0.972 |                  1   |               0.834 |             4.302 |        4.309 |
| relevance order       |         0.972 |                  0.7 |               0.653 |             6.165 |        4.048 |
