# RU22Fact claim-verification baselines

Test split, macro-F1 over Supported, Refuted and NEI; per language over the labels present (Ukrainian and Russian have no NEI claims); 95% bootstrap interval over claims.

| Input | Test macro-F1 [95% CI] | Accuracy | en | uk | ru | zh | Accuracy at 50% coverage |
|---|---|---|---|---|---|---|---|
| Majority (Supported) | 0.268 | 0.674 | | | | | |
| claim | 0.614 [0.594, 0.633] | 0.779 | 0.584 | 0.868 | 0.950 | 0.572 | 0.905 |
| evidence | 0.677 [0.655, 0.696] | 0.858 | 0.631 | 0.989 | 0.995 | 0.696 | 0.980 |
| claim+evidence | 0.679 [0.659, 0.698] | 0.859 | 0.631 | 0.989 | 0.995 | 0.713 | 0.981 |
| evidence format only (no words) | 0.591 [0.574, 0.609] | 0.761 | 0.543 | 0.944 | 0.931 | 0.352 | |
| claim+evidence, trained on English only | | | 0.641 | 0.236 | 0.927 | 0.503 | |
