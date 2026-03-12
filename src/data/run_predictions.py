import pandas as pd
import numpy as np
import torch
import json
import time
import sys
import os

# ── Add current folder to path so we can import our files ────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fc_mt_lstm_pytorch import FC_MT_LSTM, FairnessConstrainedLoss
from fairness_metrics import FairnessEvaluator

print("="*60)
print("FC-MT-LSTM PREDICTION SCRIPT")
print("="*60)

# ── Config ────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 3
HIDDEN_DIM      = 128
EPOCHS          = 50
BATCH_SIZE      = 32
LAMBDA          = 0.5
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GROUP_MAP   = {'SC': 0, 'ST': 1, 'Women': 2, 'Children': 3}
GROUP_NAMES = {0: 'SC', 1: 'ST', 2: 'Women', 3: 'Children'}
GROUP_COLORS = {
    'SC': '#64B5F6', 'ST': '#81C784',
    'Women': '#FF7043', 'Children': '#FFB74D'
}

print(f"\nDevice: {DEVICE}")

# ── Load data ─────────────────────────────────────────────────────────
print("\nLoading data...")
train_df = pd.read_csv('train_data.csv')
test_df  = pd.read_csv('test_data.csv')
print(f"Train: {len(train_df)} rows")
print(f"Test:  {len(test_df)} rows")

# ── Prepare features ──────────────────────────────────────────────────
print("\nPreparing features...")

from sklearn.preprocessing import LabelEncoder, StandardScaler

def prepare_features(df, encoders=None, scaler=None, feature_cols=None, fit=False):
    df = df.copy()

    # encode categorical columns
    cat_cols = ['state_name', 'district_name', 'protected_group']
    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in cat_cols:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x: int(le.transform([x])[0]) if x in le.classes_ else -1
            )

    # feature columns
    if fit:
        exclude = ['total_crimes', 'year', 'state_code',
                   'district_code', 'id', 'registration_circles']
        feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = df['total_crimes'].values

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, encoders, scaler, feature_cols

X_train, y_train, encoders, scaler, feature_cols = prepare_features(
    train_df, fit=True
)
X_test, y_test, _, _, _ = prepare_features(
    test_df, encoders=encoders, scaler=scaler,
    feature_cols=feature_cols, fit=False
)

print(f"Features: {len(feature_cols)}")
print(f"Train shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")

# ── Build sequences ───────────────────────────────────────────────────
print("\nBuilding sequences...")

def build_sequences(df, X, y, seq_len=3):
    sequences, targets, groups, meta = [], [], [], []

    for (state, district, group), gdf in df.groupby(
        ['state_name', 'district_name', 'protected_group']
    ):
        idx = gdf.index.tolist()
        if len(idx) < seq_len:
            continue
        for i in range(len(idx) - seq_len + 1):
            window = idx[i:i + seq_len]
            sequences.append(X[window])
            targets.append(y[window[-1]])
            groups.append(GROUP_MAP.get(group, 0))
            meta.append({
                'state':    state,
                'district': district,
                'group':    group,
                'year':     int(df.loc[window[-1], 'year'])
            })

    return (np.array(sequences), np.array(targets),
            np.array(groups), meta)

# reset index so positional indexing works
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

X_seq_tr, y_seq_tr, g_seq_tr, _    = build_sequences(train_df, X_train, y_train)
X_seq_te, y_seq_te, g_seq_te, meta = build_sequences(test_df,  X_test,  y_test)

print(f"Train sequences: {len(X_seq_tr)}")
print(f"Test sequences:  {len(X_seq_te)}")

# ── Build DataLoaders ─────────────────────────────────────────────────
from torch.utils.data import TensorDataset, DataLoader

def make_loader(X, y, g, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y).unsqueeze(1),
        torch.LongTensor(g)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_seq_tr, y_seq_tr, g_seq_tr, BATCH_SIZE)
test_loader  = make_loader(X_seq_te, y_seq_te, g_seq_te, BATCH_SIZE, shuffle=False)

# ── Build model ───────────────────────────────────────────────────────
print("\nBuilding FC-MT-LSTM model...")
n_features = X_seq_tr.shape[2]
model      = FC_MT_LSTM(input_dim=n_features, hidden_dim=HIDDEN_DIM).to(DEVICE)
criterion  = FairnessConstrainedLoss(lambda_fairness=LAMBDA)
optimizer  = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ── Train ─────────────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...")
print("(This will take 8-15 minutes)\n")

start_time = time.time()
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_X, batch_y, batch_g in train_loader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        batch_g = batch_g.to(DEVICE)

        optimizer.zero_grad()
        preds, _ = model(batch_X, batch_g)
        loss, mse, fair = criterion(preds, batch_y, batch_g)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        avg = epoch_loss / len(train_loader)
        elapsed = (time.time() - start_time) / 60
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg:.4f} | "
              f"Time: {elapsed:.1f}min")

training_time = time.time() - start_time
print(f"\n✓ Training done in {training_time/60:.1f} minutes")

# ── Predict ───────────────────────────────────────────────────────────
print("\nGenerating predictions...")
model.eval()
all_preds, all_actual, all_groups = [], [], []

with torch.no_grad():
    for batch_X, batch_y, batch_g in test_loader:
        batch_X = batch_X.to(DEVICE)
        preds, _ = model(batch_X, batch_g.to(DEVICE))
        all_preds.extend(preds.cpu().numpy().flatten())
        all_actual.extend(batch_y.numpy().flatten())
        all_groups.extend(batch_g.numpy())

all_preds  = np.maximum(all_preds, 0)
all_actual = np.array(all_actual)
all_groups = np.array(all_groups)

# ── Metrics ───────────────────────────────────────────────────────────
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae  = mean_absolute_error(all_actual, all_preds)
rmse = np.sqrt(mean_squared_error(all_actual, all_preds))
r2   = r2_score(all_actual, all_preds)

print(f"\nOverall Results:")
print(f"  MAE:  {mae:.2f}  (paper: 6.54)")
print(f"  RMSE: {rmse:.2f} (paper: 16.05)")
print(f"  R²:   {r2:.4f} (paper: 0.9922)")

# per group metrics
print(f"\nPer-Group Results:")
group_metrics = {}
for gid, gname in GROUP_NAMES.items():
    mask = all_groups == gid
    if mask.sum() == 0:
        continue
    g_mae  = mean_absolute_error(all_actual[mask], all_preds[mask])
    g_rmse = np.sqrt(mean_squared_error(all_actual[mask], all_preds[mask]))
    g_r2   = r2_score(all_actual[mask], all_preds[mask])
    group_metrics[gname] = {
        'mae': round(float(g_mae), 2),
        'rmse': round(float(g_rmse), 2),
        'r2': round(float(g_r2), 4),
        'count': int(mask.sum()),
        'color': GROUP_COLORS[gname]
    }
    print(f"  {gname:<10} MAE={g_mae:.2f}  RMSE={g_rmse:.2f}  R²={g_r2:.4f}")

maes = [v['mae'] for v in group_metrics.values()]
fairness_ratio = round(max(maes) / min(maes), 2) if min(maes) > 0 else 0
fairness_gap   = round(max(maes) - min(maes), 2)
print(f"\n  Fairness Ratio: {fairness_ratio} (paper: 1.99)")
print(f"  Fairness Gap:   {fairness_gap}  (paper: 12.61)")

# ── Build state-level predictions ─────────────────────────────────────
state_preds = {}
for i, m in enumerate(meta):
    s = m['state']
    if s not in state_preds:
        state_preds[s] = {'actual': [], 'predicted': []}
    state_preds[s]['actual'].append(float(all_actual[i]))
    state_preds[s]['predicted'].append(float(all_preds[i]))

state_preds_list = [
    {
        'state':     s,
        'actual':    round(np.mean(v['actual']), 1),
        'predicted': round(np.mean(v['predicted']), 1)
    }
    for s, v in sorted(state_preds.items())
]

# ── Top 100 sample records ─────────────────────────────────────────────
top_idx = np.argsort(all_actual)[::-1][:100]
samples = [
    {
        'state':     meta[i]['state'],
        'district':  meta[i]['district'],
        'group':     GROUP_NAMES[all_groups[i]],
        'year':      meta[i]['year'],
        'actual':    round(float(all_actual[i]), 1),
        'predicted': round(float(all_preds[i]), 1)
    }
    for i in top_idx
]

# ── Save predictions.json ─────────────────────────────────────────────
output = {
    'model': 'FC-MT-LSTM',
    'training_time_min': round(training_time / 60, 1),
    'overall': {
        'mae':  round(float(mae), 2),
        'rmse': round(float(rmse), 2),
        'r2':   round(float(r2), 4),
        'fairness_ratio': fairness_ratio,
        'fairness_gap':   fairness_gap,
    },
    'group_metrics': group_metrics,
    'state_preds':   state_preds_list,
    'samples':       samples,
}

with open('predictions.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✅ predictions.json saved!")
print(f"   {len(state_preds_list)} states · {len(samples)} sample records")
print(f"   Training time: {training_time/60:.1f} minutes")