# E3 Injected errors on IU reports (14695 erroneous reports, 14695 with a single injected error)

**(a, c) Visibility of each injected error to View 1, and Algorithm 2 routing**

| etype1 | n | v1_changed | mention_lost | mention_gained | hedge_gained | to_review |
|---|---|---|---|---|---|---|
| hedge | 1077 | 0.888 | 0.502 | 0.000 | 0.775 | 0.112 |
| insertion | 3913 | 0.504 | 0.000 | 0.503 | 0.001 | 0.000 |
| omission | 1077 | 0.502 | 0.502 | 0.000 | 0.000 | 0.502 |
| other | 3891 | 0.017 | 0.014 | 0.000 | 0.003 | 0.014 |
| side confusion | 1362 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| spelling error | 3375 | 0.049 | 0.037 | 0.000 | 0.000 | 0.037 |

**(b) Cause from the rule over (r, r') vs injected type mapped to the taxonomy**

| injected (mapped) | faithful | omission | hedge | style | conflict |
|---|---|---|---|---|---|
| conflict | 0 | 0 | 2 | 3303 | 1970 |
| hedge | 0 | 121 | 835 | 121 | 0 |
| omission | 0 | 541 | 0 | 536 | 0 |
| style | 0 | 180 | 12 | 7074 | 0 |

* recall per mapped cause: conflict 0.373, hedge 0.775, omission 0.502, style 0.974
* injected conflicts (insertion, side confusion) auto-corrected without review: 1.000
