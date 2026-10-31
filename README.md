# Network Intrusion Detection: GDG on Campus AI/ML Challenge

## The problem and why it's tricky
Classify network flows as Benign, Availability, Reconnaissance or Spoofing, scored by Macro F1. The test set contains traffic types that train never shows. So a model that looks great on a random split (about 97%) can still fail on test. I treated this as a distribution-shift problem, not a model-picking problem.

## Methodology and reasoning
**1. Validation.** *What:* I grouped similar flows into clusters within each class and held whole clusters out in each fold. *Why:* a random split puts near-copies of every flow in both train and validation, so it can't show how a model handles unseen traffic. This is a tougher test than the real one (scores around 0.54, vs 0.84 on the public board), so I only used it to compare models with each other.

**2. Features.** *What:* I added total packets, forward/backward packet ratio and SYN fraction, and dropped `id.orig_p`. *Why:* the ratios don't depend on traffic volume. `id.orig_p` looks like a random client port, so it only identifies specific captures, and dropping it helped my validation slightly (0.534 to 0.543).

**3. Model.** *What:* LightGBM with shallow trees and strong regularization, balanced class weights, and 5 random seeds averaged. *Why:*
- LightGBM clearly beat Random Forest and ExtraTrees in my validation (0.54 vs about 0.40).
- Strong regularization stops the model memorizing quirks of the training traffic. Going from a lighter model to this one took the public score from 0.818 to 0.845.
- Balanced weights because train is 94.5% Availability and Macro F1 counts all four classes equally.
- Averaging seeds smooths out random variation.

**4. Pseudo-labeling.** *What:* I added test rows the model was very sure about, using its own predictions as labels, then retrained. *Why:* it lets the model see the test traffic's feature patterns. It uses no hand labels and no hidden labels.

## Main finding: the model learned that "no reply" means a port scan
*How I found it:* every model scored about 0.845 on the public board, even when I changed 28% of the test predictions. So the public set didn't contain the rows my models disagreed on, and I compared test features against train to find them.

About 31% of test is one attack: a SYN flood on port 21. In train it is always labeled Availability (79,698 rows), and every flow got a reply. In test, 9,159 of them got none. In train, "no reply" usually meant a port scan (Reconnaissance), so the model labeled those rows Reconnaissance.

*Fix:* I made copies of the training attack flows with the reply information blanked out, so the model recognizes the attack by what the attacker sends. *Check:* on held-out train flows with replies hidden, the old model found 0% of the attacks and the fixed one found 100%. `python solution.py --check` reproduces this.

## Final submissions and why
| File | Description | Public LB |
|---|---|---|
| `submission_bag.csv` | regularized 5-seed LightGBM | 0.84486 |
| `submission_pl.csv` | bag + pseudo-labeling | 0.84475 |
| `submission_augA.csv` | bag + the reply fix | 0.84335 |

I kept bag and PL as my two finals. They label those 9,159 rows differently (bag: Reconnaissance, PL: Availability), and the public set has almost none of those rows, so I couldn't check it there. The pair covers both outcomes. My analysis pointed to Availability. I can't fully explain why PL handles those rows that way, so the reply fix is the version I can justify directly.

## What I tried and rejected
- **Removing features that differ between train and test:** public score fell to 0.730, so those features carried real signal.
- **Smoothing predictions across similar test rows, and correcting for class mix:** no gain in validation. The model's errors were whole groups of rows, not scattered ones.
- **Other models (CatBoost, XGBoost, logistic regression) and blends:** no better, and they mostly agree with LightGBM.

## Limitations
- About 2,000 test rows (port 5353, mDNS) look unlike anything in train, so I couldn't resolve them.
- My validation is only a rough proxy for the real shift.
- I couldn't work out what the top public scores did differently.

## Run it
```
pip install -r requirements.txt
python solution.py --data-dir <folder with train.csv and test.csv> --out-dir out --check
```
`nids_final_notebook.ipynb` has the full story and experiments.

## Tools
I used an AI assistant for code and analysis help, which the rules allow. I went through the code and can explain each step.
