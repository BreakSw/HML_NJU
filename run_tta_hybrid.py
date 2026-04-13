import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from tableshift import get_dataset

# ====================== 配置 ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
SELF_TRAIN_EPOCHS = 3
CONF_THRESH = 0.9

# ====================== 加载数据集 ======================
print("加载数据集...")
dataset = get_dataset("adult", cache_dir="/app/data/TableShift Dataset")
train_loader = dataset.get_dataloader(split="train", batch_size=BATCH_SIZE, shuffle=True)
test_loader = dataset.get_dataloader(split="test", batch_size=BATCH_SIZE)

# ====================== 模型 ======================
x, y, _ = next(iter(train_loader))
in_dim = x.shape[1]
num_classes = 2

class MLP(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, n_cls)
        )
    def forward(self, x):
        return self.model(x)

model = MLP(in_dim, num_classes).to(DEVICE)

# ====================== 预训练 ======================
opt = torch.optim.Adam(model.parameters(), lr=LR)
ce = nn.CrossEntropyLoss()

print("\n开始预训练...")
for ep in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y, _ in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE).long()
        opt.zero_grad()
        loss = ce(model(x), y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {ep+1:2d} | Loss: {total_loss/len(train_loader):.3f}")

# ====================== 测试函数 ======================
def test(loader, model):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(DEVICE)
            pred = model(x).argmax(1)
            preds.append(pred.cpu())
            labels.append(y)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return accuracy_score(labels, preds)

acc_erm = test(test_loader, model)
print(f"\n===== ERM baseline 准确率: {acc_erm:.4f} =====")

# ====================== 🚀 混合增强+自训练 ======================
print("\n开始混合增强+自训练...")
model_hybrid = MLP(in_dim, num_classes).to(DEVICE)
model_hybrid.load_state_dict(model.state_dict())
opt_hybrid = torch.optim.AdamW(model_hybrid.parameters(), lr=1e-4, weight_decay=1e-4)

for ep in range(SELF_TRAIN_EPOCHS):
    model_hybrid.train()
    total_loss = 0
    for x, y, _ in test_loader:
        x = x.to(DEVICE)
        # 特征增强：加噪声+掩码
        x_aug = x + torch.randn_like(x) * 0.008
        x_aug = x_aug * torch.bernoulli(torch.ones_like(x_aug) * 0.99)
        opt_hybrid.zero_grad()
        logits = model_hybrid(x_aug)
        prob = torch.softmax(logits, dim=1)
        max_prob, pseudo_y = torch.max(prob, dim=1)
        mask = max_prob > CONF_THRESH
        if mask.sum() == 0:
            continue
        loss = ce(logits[mask], pseudo_y[mask])
        loss.backward()
        opt_hybrid.step()
        total_loss += loss.item()
    print(f"混合训练 Epoch {ep+1} | 平均损失: {total_loss/len(test_loader):.4f}")

acc_hybrid = test(test_loader, model_hybrid)

# ====================== 最终结果 ======================
print("\n" + "="*60)
print("             🎯 混合增强+自训练 最终结果")
print("="*60)
print(f"ERM baseline      |  {acc_erm:.4f}")
print(f"混合增强+自训练  |  {acc_hybrid:.4f}")
print(f"提升              |  +{acc_hybrid - acc_erm:.4f}")
print("="*60)