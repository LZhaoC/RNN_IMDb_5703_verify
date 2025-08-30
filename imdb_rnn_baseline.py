# imdb_rnn_baseline.py
import time, random, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd
from collections import Counter     # 用于词表构建

SEED = 42 # 固定随机种子以确保可复现性
random.seed(SEED); torch.manual_seed(SEED)

# ==== 设备选择（支持 Apple Silicon MPS）====
def get_device():
    if torch.backends.mps.is_available(): 
        return torch.device("mps")# 优先使用 Apple MPS
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") # 否则 CUDA 或 CPU

# ==== 简单 tokenizer / 词表 / 编码 ====
def tokenize(x):
    return x.lower().strip().split()   # 简单分词：小写+空格切分

def build_vocab(texts, max_size=20000, min_freq=2):
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))
    itos = ["<pad>", "<unk>"]       # index to string: 0=<pad>, 1=<unk>
    for w, f in cnt.most_common():
        if len(itos) >= max_size: break
        if f >= min_freq:
            itos.append(w)
    stoi = {w:i for i,w in enumerate(itos)}   # string to index
    return stoi, itos

def encode(text, stoi, max_len=256):
    ids = [stoi.get(tok, 1) for tok in tokenize(text)]  # 1=<unk>  # OOV 映射到 <unk>=1
    ids = ids[:max_len]     # 截断到 max_len    
    return ids + [0]*(max_len - len(ids))              # 0=<pad>    # 补 pad 到固定长度

# ==== PyTorch Dataset 封装 ====
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, stoi, max_len=256):
        self.x = [encode(t, stoi, max_len) for t in texts]
        self.y = labels
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return (torch.tensor(self.x[i], dtype=torch.long),
                torch.tensor(self.y[i], dtype=torch.float32))

# ==== 模型：Embedding -> LSTM -> FC ====
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden=256, num_layers=1,
                 bidir=True, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)    # 词向量层
        self.rnn = nn.LSTM(emb_dim, hidden, num_layers=num_layers,   # LSTM 编码
                           batch_first=True, bidirectional=bidir,
                           dropout=0.0 if num_layers==1 else dropout)   
        out_dim = hidden * (2 if bidir else 1)               # 双向则 ×2
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)                # 输出层（二分类）

    def forward(self, x):             # x: [B, T]
        emb = self.drop(self.emb(x))  # [B, T, E]
        out, (h, c) = self.rnn(emb)   # out: [B, T, H*dirs]
        feat = out[:, -1, :]          # 取最后时间步（也可换成max/mean-pool）
        feat = self.drop(feat)
        logit = self.fc(feat).squeeze(1)  # [B] # [B]，未经过 sigmoid
        return logit

# ==== 训练与评估 ====
def accuracy_from_logits(logits, y):
    preds = (torch.sigmoid(logits) > 0.5).long().cpu()
    return (preds.squeeze() == y.long().cpu()).float().mean().item()

def run_baseline():
    device = get_device()
    print("Device:", device)

    # 1) 数据
    ds = load_dataset("imdb")
    train_texts = [x["text"] for x in ds["train"]]
    train_labels = [int(x["label"]) for x in ds["train"]]
    test_texts  = [x["text"] for x in ds["test"]]
    test_labels = [int(x["label"]) for x in ds["test"]]

    # 划分 train/val = 90/10
    n = len(train_texts)
    idx = list(range(n)); random.shuffle(idx)
    cut = int(0.9*n)
    tr_idx, va_idx = idx[:cut], idx[cut:]
    tr_texts = [train_texts[i] for i in tr_idx]; tr_labels = [train_labels[i] for i in tr_idx]
    va_texts = [train_texts[i] for i in va_idx]; va_labels = [train_labels[i] for i in va_idx]

    # 词表
    stoi, itos = build_vocab(tr_texts, max_size=20000, min_freq=2)

    # 数据集 & DataLoader
    train_ds = IMDBDataset(tr_texts, tr_labels, stoi, max_len=256)
    val_ds   = IMDBDataset(va_texts, va_labels, stoi, max_len=256)
    test_ds  = IMDBDataset(test_texts, test_labels, stoi, max_len=256)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=2)

    # 2) 模型/优化
    model = RNNClassifier(vocab_size=len(itos), emb_dim=100,
                          hidden=256, num_layers=1, bidir=True, dropout=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 3) 训练（早停）
    best_val_acc, best_val_loss, best_epoch = 0.0, 1e9, -1
    best_state = None
    start_time = time.time()
    patience, bad_epochs = 3, 0

    for epoch in range(1, 21):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logit = model(x)
            loss = criterion(logit, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        # 验证
        model.eval(); total, correct, vloss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logit = model(x)
                vloss_sum += criterion(logit, y).item() * x.size(0)
                preds = (torch.sigmoid(logit) > 0.5).long()
                correct += (preds.squeeze().cpu() == y.long().cpu()).sum().item()
                total += x.size(0)
        val_acc = correct / total
        val_loss = vloss_sum / total

        if val_acc > best_val_acc:
            best_val_acc, best_val_loss, best_epoch = val_acc, val_loss, epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(f"Epoch {epoch:02d} | val_acc={val_acc:.4f} | val_loss={val_loss:.4f}")
        if bad_epochs >= patience: break

    # 载入最佳
    model.load_state_dict(best_state, strict=True)
    model.to(device); model.eval()

    # 测试
    total, correct, tloss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            logit = model(x)
            tloss_sum += criterion(logit, y).item() * x.size(0)
            preds = (torch.sigmoid(logit) > 0.5).long()
            correct += (preds.squeeze().cpu() == y.long().cpu()).sum().item()
            total += x.size(0)
    test_acc = correct / total
    test_loss = tloss_sum / total
    train_time = round(time.time() - start_time, 1)

    # 输出 Performance Table
    best_row = {
        "Best Val Epoch": best_epoch,
        "Best Val Acc": round(best_val_acc, 4),
        "Best Val Loss": round(best_val_loss, 4),
        "Test Acc": round(test_acc, 4),
        "Test Loss": round(test_loss, 4),
        "Train Time (s)": train_time
    }
    df = pd.DataFrame([best_row])
    print("\nPerformance Table (Base RNN Model)")
    print(df.to_string(index=False))

#if __name__ == "__main__":
 #   run_baseline()

# ====== NEW: LoRA 适配器与 LoRA 版 RNN 分类器 ======
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=8, alpha=1.0, bias=True, freeze_base=True):
        super().__init__()
        self.in_f, self.out_f, self.r, self.alpha = in_f, out_f, r, alpha
        # 基础权重（可选冻结）
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if freeze_base:
            self.weight.requires_grad = False
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        # LoRA 低秩增量 ΔW = A @ B
        self.A = nn.Parameter(torch.zeros(out_f, r))
        self.B = nn.Parameter(torch.zeros(r, in_f))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):  # x: [B, in_f]
        W_eff = self.weight + self.alpha * (self.A @ self.B)
        y = x @ W_eff.t()
        if self.bias is not None:
            y = y + self.bias
        return y

class RNNClassifierLoRA(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden=256, num_layers=1,
                 bidir=True, dropout=0.3, lora_r=8, lora_alpha=1.0,
                 freeze_base_fc=True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hidden, num_layers=num_layers,
                           batch_first=True, bidirectional=bidir,
                           dropout=0.0 if num_layers==1 else dropout)
        out_dim = hidden * (2 if bidir else 1)
        self.drop = nn.Dropout(dropout)
        # 仅在分类头使用 LoRA（最稳妥）
        self.fc = LoRALinear(out_dim, 1, r=lora_r, alpha=lora_alpha, bias=True,
                             freeze_base=freeze_base_fc)

    def forward(self, x):
        emb = self.drop(self.emb(x))
        out, _ = self.rnn(emb)
        feat = self.drop(out[:, -1, :])
        logit = self.fc(feat).squeeze(1)
        return logit

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ====== NEW: 运行 LoRA 实验并输出表格（含 Trainable Params） ======
def run_lora_experiment():
    device = get_device()
    ds = load_dataset("imdb")
    train_texts = [x["text"] for x in ds["train"]]
    train_labels = [int(x["label"]) for x in ds["train"]]
    test_texts  = [x["text"] for x in ds["test"]]
    test_labels = [int(x["label"]) for x in ds["test"]]

    n = len(train_texts)
    idx = list(range(n)); random.shuffle(idx)
    cut = int(0.9*n)
    tr_idx, va_idx = idx[:cut], idx[cut:]
    tr_texts = [train_texts[i] for i in tr_idx]; tr_labels = [train_labels[i] for i in tr_idx]
    va_texts = [train_texts[i] for i in va_idx]; va_labels = [train_labels[i] for i in va_idx]

    stoi, itos = build_vocab(tr_texts, max_size=20000, min_freq=2)

    train_ds = IMDBDataset(tr_texts, tr_labels, stoi, max_len=256)
    val_ds   = IMDBDataset(va_texts, va_labels, stoi, max_len=256)
    test_ds  = IMDBDataset(test_texts, test_labels, stoi, max_len=256)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=2)

    model = RNNClassifierLoRA(vocab_size=len(itos), emb_dim=100,
                              hidden=256, num_layers=1, bidir=True, dropout=0.3,
                              lora_r=8, lora_alpha=1.0, freeze_base_fc=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    # 只优化可训练参数（LoRA + bias）
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    best_val_acc, best_val_loss, best_epoch = 0.0, 1e9, -1
    best_state = None
    start_time = time.time()
    patience, bad_epochs = 3, 0

    for epoch in range(1, 21):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval(); total, correct, vloss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vloss_sum += criterion(logits, y).item() * x.size(0)
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds.squeeze().cpu() == y.long().cpu()).sum().item()
                total += x.size(0)
        val_acc = correct / total
        val_loss = vloss_sum / total

        if val_acc > best_val_acc:
            best_val_acc, best_val_loss, best_epoch = val_acc, val_loss, epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience: break

    model.load_state_dict(best_state, strict=True)
    model.to(device); model.eval()
    total, correct, tloss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            tloss_sum += criterion(logits, y).item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds.squeeze().cpu() == y.long().cpu()).sum().item()
            total += x.size(0)

    test_acc = correct / total
    test_loss = tloss_sum / total
    train_time = round(time.time() - start_time, 1)

    best_row = {
        "Best Val Epoch": best_epoch,
        "Best Val Acc": round(best_val_acc, 4),
        "Best Val Loss": round(best_val_loss, 4),
        "Test Acc": round(test_acc, 4),
        "Test Loss": round(test_loss, 4),
        "Train Time (s)": train_time,
        "Trainable Params": count_trainable(model)
    }
    df = pd.DataFrame([best_row])
    print("\nPerformance Table (RNN + LoRA)")
    print(df.to_string(index=False))

# 作为单独运行入口：
if __name__ == "__main__":
    run_lora_experiment()
   
