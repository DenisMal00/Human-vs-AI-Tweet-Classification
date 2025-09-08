import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import html
import string
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from itertools import product

# === Funzioni base ===
def clean_text_base(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def count_punctuation(text):
    return sum(1 for c in text if c in string.punctuation)

def repetition_ratio(text):
    words = text.split()
    if not words: return 0
    return 1 - len(set(words)) / len(words)

def extract_features(df, text_col='text_clean'):
    df = df.copy()
    df['has_url'] = df[text_col].str.contains(r'http\S+').astype(int)
    df['has_mention'] = df[text_col].str.contains(r'@\w+').astype(int)
    df['has_hashtag'] = df[text_col].str.contains(r'#\w+').astype(int)
    df['num_punctuations'] = df[text_col].apply(count_punctuation)
    df['num_uppercase_words'] = df[text_col].apply(lambda x: sum(1 for w in x.split() if w.isupper()))
    df['num_chars'] = df[text_col].str.len()
    df['num_words'] = df[text_col].apply(lambda x: len(x.split()))
    df['text_len_ratio'] = df.apply(lambda r: r['num_chars']/r['num_words'] if r['num_words'] > 0 else 0, axis=1)
    df['repetition_ratio'] = df[text_col].apply(repetition_ratio)
    return df

def tokenize(text): return text.split()

def build_vocab(texts, min_freq=1):
    freq = defaultdict(int)
    for text in texts:
        for token in tokenize(text):
            freq[token] += 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, count in freq.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def encode(text, vocab, max_len):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens[:max_len]]
    return ids + [vocab['<PAD>']] * (max_len - len(ids))

class TweetDataset(Dataset):
    def __init__(self, inputs, extra_feats, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.extra_feats = torch.tensor(extra_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.inputs[idx], self.extra_feats[idx], self.labels[idx]

class MLPWithExtraFeatures(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, dropout1, dropout2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.extra_proj = nn.Linear(9, 16)
        self.classifier = nn.Sequential(
            nn.Linear(2 * emb_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, extra_feats):
        emb = self.embedding(x)
        mean_pool = emb.mean(dim=1)
        max_pool = emb.max(dim=1).values
        extra = F.relu(self.extra_proj(extra_feats))
        x = torch.cat([mean_pool, max_pool, extra], dim=1)
        return self.classifier(x)

# === Caricamento dati ===
train_df = pd.read_csv('files/train.csv', sep=';', quotechar='"', engine='python')
val_df = pd.read_csv('files/validation.csv', sep=';', quotechar='"', engine='python')
test_df = pd.read_csv('files/test.csv', sep=';', quotechar='"', engine='python')

for df in [train_df, val_df, test_df]:
    df['text_clean'] = df['text'].apply(clean_text_base)
train_df = extract_features(train_df)
val_df = extract_features(val_df)
test_df = extract_features(test_df)

vocab = build_vocab(train_df['text_clean'])
le = LabelEncoder()
train_labels = le.fit_transform(train_df['account.type'])
val_labels = le.transform(val_df['account.type'])
test_labels = le.transform(test_df['account.type'])

feature_cols = ['num_words', 'num_chars', 'has_url', 'has_mention', 'has_hashtag',
                'num_punctuations', 'num_uppercase_words', 'text_len_ratio', 'repetition_ratio']
scaler = StandardScaler()
train_extra = scaler.fit_transform(train_df[feature_cols])
val_extra = scaler.transform(val_df[feature_cols])
test_extra = scaler.transform(test_df[feature_cols])

# === Grid Search ===
param_grid = list(product(
    [200, 300],                        # 2 valori → embedding_dim
    [32, 64, 128],                     # 3 valori → hidden_dim
    [1e-3, 5e-5],                      # 2 valori → learning_rate
    [1e-2, 1e-3],                      # 2 valori → weight_decay
    [(0.5, 0.4), (0.4, 0.3)],          # 2 valori → dropout
    [40, 60],                          # 2 valori → max_len
    [64, 128, 256]                     # 3 valori → batch_size
))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_val_acc = 0
best_params = None
results = []

for i, (emb_dim, hid_dim, lr, wd, (do1, do2), max_len, batch_size) in enumerate(param_grid):
    print(f"\n[Trial {i+1}/{len(param_grid)}] emb_dim={emb_dim}, hidden_dim={hid_dim}, lr={lr}, wd={wd}, dropouts=({do1},{do2}), max_len={max_len}, batch_size={batch_size}")

    train_enc = [encode(t, vocab, max_len) for t in train_df['text_clean']]
    val_enc = [encode(t, vocab, max_len) for t in val_df['text_clean']]
    test_enc = [encode(t, vocab, max_len) for t in test_df['text_clean']]

    train_dataset = TweetDataset(train_enc, train_extra, train_labels)
    val_dataset = TweetDataset(val_enc, val_extra, val_labels)
    test_dataset = TweetDataset(test_enc, test_extra, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = MLPWithExtraFeatures(len(vocab), emb_dim, hid_dim, 2, do1, do2).to(device)
    class_counts = np.bincount(train_labels)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_acc = 0
    patience, counter = 10, 0

    for epoch in range(100):
        model.train()
        for xb, xf, yb in train_loader:
            xb, xf, yb = xb.to(device), xf.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, xf), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, xf, yb in val_loader:
                xb, xf, yb = xb.to(device), xf.to(device), yb.to(device)
                pred = model(xb, xf).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            counter = 0
            if acc > best_val_acc:
                torch.save(model.state_dict(), 'best_MLP.pt')
        else:
            counter += 1
            if counter >= patience:
                break

    results.append((best_acc, emb_dim, hid_dim, lr, wd, do1, do2, max_len, batch_size))
    if best_acc > best_val_acc:
        best_val_acc = best_acc
        best_params = (emb_dim, hid_dim, lr, wd, do1, do2, max_len, batch_size)

# === Valutazione finale ===
emb_dim, hid_dim, lr, wd, do1, do2, max_len, batch_size = best_params
print(f"\n✅ Best Val Acc: {best_val_acc:.4f} with params:")
print(f"embedding_dim={emb_dim}, hidden_dim={hid_dim}, lr={lr}, weight_decay={wd}, dropout=({do1},{do2}), max_len={max_len}, batch_size={batch_size}")

# Ricarica modello migliore e testa
model = MLPWithExtraFeatures(len(vocab), emb_dim, hid_dim, 2, do1, do2).to(device)
model.load_state_dict(torch.load('best_MLP.pt'))
model.eval()

all_preds, all_labels = [], []
test_enc = [encode(t, vocab, max_len) for t in test_df['text_clean']]
test_dataset = TweetDataset(test_enc, test_extra, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

with torch.no_grad():
    for xb, xf, yb in test_loader:
        xb, xf = xb.to(device), xf.to(device)
        pred = model(xb, xf).argmax(1).cpu()
        all_preds.extend(pred.tolist())
        all_labels.extend(yb.tolist())

print("\nTest Classification Report:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(pd.DataFrame(cm, index=[f"true_{l}" for l in le.classes_],
                       columns=[f"pred_{l}" for l in le.classes_]))
