# Model Card: Customer Risk Logistic Classifier

## Intended use

The model demonstrates reproducible evaluation mechanics for prioritizing voluntary retention outreach in a synthetic environment. It must not be used for eligibility, pricing, employment, credit, insurance or other consequential decisions.

## Data and training

`data.py` generates 720 fictional customer records from seed 421. Features cover tenure, recent support tickets, usage, late payments, plan and region. Labels follow a documented synthetic risk equation with controlled random noise. The pipeline creates a stratified 70/15/15 train, validation and test split. A logistic classifier is trained from zero-valued weights with fixed hyperparameters; the decision threshold is selected on validation F1 and evaluated once on test.

## Evaluation

The release report compares the model with a majority-class baseline fitted exclusively on the training partition and gates test F1, recall and lift. It reports Brier score, confusion counts and per-segment metrics for region and plan. Stable extreme-risk cases form a behavioral regression suite. Identity and shifted copies of held-out data verify both the specificity and sensitivity paths of PSI monitoring.

## Limitations

Synthetic labels do not establish real-world validity. Region is included only to make disparity analysis executable and should be reviewed or removed for a real deployment. The model has no causal interpretation, uncertainty interval, live feedback loop or external calibration study. Passing this repository's gates demonstrates software and evaluation behavior only.
