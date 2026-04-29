import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import joblib

# ─── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv('movement_gesture_data.csv')
X  = df.drop('label', axis=1).values.astype(np.float32)
y  = df['label'].values

# Show class distribution
print("Sample counts per gesture:")
for label, count in sorted(Counter(y).items()):
    print(f"  {label}: {count}")

le    = LabelEncoder()
y_enc = le.fit_transform(y)
joblib.dump(le, 'label_encoder_movement.pkl')
print(f"\nClasses: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

# Weighted sampler so unbalanced classes don't skew training
class_counts = np.bincount(y_train)
weights      = 1.0 / class_counts[y_train]
sampler      = WeightedRandomSampler(weights, len(weights))
loader       = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=32, sampler=sampler)

# ─── Model ────────────────────────────────────────────────────────────────────
class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model     = GestureNet(X_train.shape[1], len(le.classes_))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

BEST_CHECKPOINT = 'gesture_movement_best.pt'
FINAL_MODEL     = 'gesture_movement.pt'

best_acc   = 0.0
best_epoch = 0

print("\nTraining...\n")
for epoch in range(300):
    # train
    model.train()
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    # eval
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t)
            preds  = logits.argmax(1)
            acc    = (preds == y_test_t).float().mean().item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:3d}  loss={avg_loss:.4f}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc   = acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_CHECKPOINT)

# ─── Final report ─────────────────────────────────────────────────────────────
print(f"\nBest val_acc={best_acc:.3f} at epoch {best_epoch}  (saved as {BEST_CHECKPOINT})")

model.load_state_dict(torch.load(BEST_CHECKPOINT))
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(1).numpy()

print("\nPer-class report:")
print(classification_report(le.inverse_transform(y_test),
                             le.inverse_transform(preds)))

# Save final
torch.save(model.state_dict(), FINAL_MODEL)
joblib.dump(le, 'label_encoder_movement.pkl')
print(f"Done. {FINAL_MODEL} + label_encoder_movement.pkl ready.")