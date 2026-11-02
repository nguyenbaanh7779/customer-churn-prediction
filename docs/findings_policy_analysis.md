# Policy analysis: findings (notebook 10)

Branch `feat/policy-analysis`. Reproduce: run notebooks 07 → 09 → 10 (`python src/build_nb10.py` regenerates notebook 10). All £ are scenario estimates under Strategy v2 (base: d=10%, e=30%, m=40%, r=40%, contact £0.50), per week, test = Jul–Sep 2011.

1. **Scores in `output/` were an older model** (`lr_full_c0.01_h90`). Rerunning notebooks 07+09 on main reproduces `lr_extended_c0.01_weighted_h90`; probabilities differ by up to 0.25 (mean 0.05). Use `data/results/churn_scores_h90.parquet` from the rerun.
2. **Churn-only and recency share only 33–38% of their top-10%.** Model-only picks: mean V £181, 84% one-time, churn 73%. Recency-only picks: mean V £396, repeat buyers, churn 53%. This confirms the "right churners, small money" explanation.
3. **Calibration on validation does not transfer.** Churn rate is 53% (val) vs 40% (test); isotonic/Platt fitted on validation worsen test Brier (0.197 → 0.199). The train-only model is slightly under-confident on test (mean p 0.37 vs 0.40), the train+val refit over-confident (0.47). Level moves with season; ranking is stable (AUC 0.75–0.78).
4. **Segment actions differ from their names.** Offer pays for At Risk, Hibernating, Potential. Champions and Loyal lose margin. Can't Lose (churn only 29–39%) is not worth a discount.
5. **Policy ranking (test, base, K=20%, £/week):** P6 value-aware 7,753 > P5 risk-only 5,463 > P3 recency 4,911 > P2 random 1,124. P6 wins all 27 cells of the d/e × m × r grid, and is picked on validation in 14/15 scenario×K cells. Risk-only loses to recency at K=10%.
6. **Blanket is a mixed reference.** It beats K=20% targeting in the Light scenario (cheap discount) but loses £22k/week in the Deep scenario; P6 beats it in 16/27 grid cells at K=20%. Per contact P6 earns ~£16 vs ~£1.8.
7. **Wholesale influence:** ranking holds when the top-1% customers are dropped. Uncapped V lets recency overtake risk-only (5,840 vs 5,463), so the P99 cap matters for that comparison.
8. **Multi-week double counting:** with a 4-week suppression the weekly margin falls ~60% (P6 7,753 → 3,067). Do not sum weekly figures into a campaign total.

## Rolling-origin backtest (notebook 11)

Four expanding-window folds with a 90-day embargo, same recipe refit each time: A Dec 2010–Jan 2011 (churn 61%), B Feb–Mar 2011 (54%), C Apr–May 2011 (46%), D Jul–Sep 2011 (40%, official test). Fold A trains on only 14 weeks.

9. **Ranking holds in every season.** P6 value-aware beats recency in 12/12 fold × scenario cells and risk-only in 11/12. It is the best of random/recency/risk-only/P6 in 11/12 (the exception: Deep scenario in the Jul–Sep fold, where risk-only and a segment rule edge ahead).
10. **Model ranking quality is stable, probability level is not.** ROC-AUC 0.76–0.78 in all folds; Lift@10% beats recency in every fold (1.45 vs 1.27 in peak-churn fold A up to 1.89 vs 1.57 in D). But mean predicted p is 0.39–0.47 across folds while the actual churn rate is 0.61 → 0.40: the model has no seasonal signal, so it under-predicts in peak churn and over-predicts in summer. This is the real calibration issue (not a one-off offset); needs a modeling decision (calendar/seasonal feature or per-period base-rate adjustment).
11. **Blanket flips with the season.** Base scenario, K=20%: blanket total is higher than P6 in the three higher-churn folds (A 36.7k vs 19.7k £/week) but P6 earns 2–3× more per contact (A £31 vs £12; D £17 vs £1.8). In the Deep scenario blanket loses money in folds B, C, D (−£3k to −£22k/week) while P6 stays positive in all folds. The right reach depends on churn level: a higher K (or a segment rule) pays in peak-churn weeks; a tight K when churn is low.
12. **Segment actions are mostly stable, with one season-dependent exception.** At Risk, Hibernating, Potential earn a positive margin per contact in all folds; Champions lose in all folds; Loyal is about zero or negative. **Can't Lose** earns a small positive margin in folds A–C (high churn) and a loss in D: offer only when churn is high; this corrects the earlier "no discount" note, which was based on the low-churn test quarter alone.
13. **Segment rule (P4b) is a strong simple baseline**: in the Base scenario it earns more total £/week than P6 in all four folds, but contacts 3× more customers (per contact it earns less in every fold). Compare at equal budget before presenting it as better.

## Margin decomposition and break-even margin (notebooks 10 §10.5b, 11 §11.5)

The dataset has revenue only (no cost or margin): `m` is an assumption, so each policy is also expressed as the **break-even gross margin** m* at which its net margin is zero (net is linear in m).

14. **Break-even margin, K=20%, test, Base (d=10%, e=30%):** risk-only 0.18, value-aware 0.20, P4b 0.21, recency 0.22, random 0.34, blanket 0.35. Targeted policies pay off if gross margin exceeds ~20%; blanket needs >35%. Deep scenario (d=20%): value-aware 0.33, blanket 0.57. Across the four backtest folds value-aware is below recency in every fold (A 0.13 vs 0.16, D 0.21 vs 0.22).
15. **Where the margin comes from (fold D test, £/week, Base):** value-aware preserves ~£39k of expected revenue (e·V of captured churners) and gives away £3.7k of discount to would-be buyers; blanket preserves £97k but gives away £23k to buyers. Revenue preserved is e·V under the assumption that a response to an offer is a retained purchase, not an observed lift.
