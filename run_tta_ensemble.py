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
ENSEMBLE_NUM = 3  # 集成模型数量

# ====================== 加载数据集 ======================
print("加载数据集...")
dataset = get_dataset("adult", cache_dir="/app/data/TableShift Dataset")
train_loader = dataset.get_dataloader(split="train", batch_size=BATCH_SIZE, shuffle=True)
test_loader = dataset.get_dataloader(split="test", batch_size=BATCH_SIZE)

# ====================== 模型定义 ======================
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

# ====================== 训练多个模型 ======================
models = []
for i in range(ENSEMBLE_NUM):
    print(f"\n===== 训练第 {i+1}/{ENSEMBLE_NUM} 个模型 =====")
    model = MLP(in_dim, num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
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
    models.append(model)

# ====================== 单模型测试（ERM baseline）======================
def test_single(loader, model):
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

acc_erm_list = [test_single(test_loader, m) for m in models]
acc_erm = np.mean(acc_erm_list)
print(f"\n===== ERM baseline 平均准确率: {acc_erm:.4f} =====")

# ====================== 🚀 集成预测（核心涨点）======================
print("\n开始集成预测...")
def test_ensemble(loader, models):
    for m in models:
        m.eval()
    all_probs = []
    labels = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(DEVICE)
            labels.append(y)
            # 收集所有模型的预测概率
            probs_list = [torch.softmax(m(x), dim=1) for m in models]
            # 概率平均
            avg_prob = torch.stack(probs_list).mean(dim=0)
            all_probs.append(avg_prob.cpu())
    all_probs = torch.cat(all_probs).numpy()
    labels = torch.cat(labels).numpy()
    preds = np.argmax(all_probs, axis=1)
    return accuracy_score(labels, preds)

acc_ensemble = test_ensemble(test_loader, models)

# ====================== 最终结果 ======================
print("\n" + "="*60)
print("             🎯 模型集成 最终结果")
print("="*60)
print(f"ERM baseline（平均）|  {acc_erm:.4f}")
print(f"3模型集成          |  {acc_ensemble:.4f}")
print(f"提升                |  +{acc_ensemble - acc_erm:.4f}")
print("="*60)