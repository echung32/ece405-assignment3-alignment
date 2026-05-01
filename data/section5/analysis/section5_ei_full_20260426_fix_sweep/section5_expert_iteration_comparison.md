# Section 5 Expert Iteration Comparison

## Key Takeaways

- The best EI configuration was `group 8, db 1024, epochs 2, EI 5` with best validation accuracy 0.5117 at EI step 5.
- The weakest EI configuration was `group 4, db 1024, epochs 2, EI 5` with best validation accuracy 0.4277, so the sweep spans 0.0840 accuracy.
- Compared against the best Section 4 SFT baseline (`2e-5` at 0.5064), the strongest EI run improved by 0.0053.

## Required Discussion

Relative to the best Section 4 SFT checkpoint, expert iteration delivers a small gain: the top EI run reaches 0.5117 versus 0.5064 for SFT.
Across EI steps, the highest-performing configuration peaks at EI step 5, while later steps do not produce a uniformly monotonic improvement across all settings, so the sweep quality depends on the configuration rather than EI step alone.

## Run Table

| Run | Best Accuracy | Final Accuracy | Best EI Step | Entropy At Best |
| --- | ---: | ---: | ---: | ---: |
| `group 8, db 1024, epochs 2, EI 5` | 0.5117 | 0.5117 | 5 | 0.2250 |
| `group 8, db 2048, epochs 1, EI 5` | 0.5039 | 0.4766 | 4 | 0.2645 |
| `group 4, db 512, epochs 1, EI 5` | 0.4346 | 0.3887 | 2 | 0.3561 |
| `group 4, db 1024, epochs 2, EI 5` | 0.4277 | 0.4062 | 3 | 0.2918 |
