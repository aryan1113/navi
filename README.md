## Navi PD assignment (quick run)

This repo trains a simple PD model on `type='train'` and scores PDs for `type='oot'`.

### Setup (use only `navivenv`)

```bash
source navivenv/bin/activate
python -m pip install -r requirements.txt
```

### Train

Logistic Regression (default):

```bash
source navivenv/bin/activate
python -m src.train --model lr
```

Or Random Forest:

```bash
source navivenv/bin/activate
python -m src.train --model rf
```

Outputs:
- `models/pd_model.pkl`
- `models/threshold.json`
- `models/metrics.json`

### Predict (OOT submission)

```bash
source navivenv/bin/activate
python -m src.predict
```

writes `outputs/oot_predictions.csv` with columns: `decision_id`, `predicted_PD`.
