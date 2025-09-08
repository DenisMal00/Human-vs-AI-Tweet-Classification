import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Config
MODEL_NAME = "bert-base-uncased"
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 0
LR         = 2e-5
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Dataset
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(texts,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=max_len,
                                   return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# 2. Model with higher dropout
class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),   # aumentato a 0.6
            nn.Linear(self.bert.config.hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

# 3. Load and preprocess data
def load_and_encode(tokenizer):
    train_df = pd.read_csv('files/train.csv',       sep=';', quotechar='"')
    val_df   = pd.read_csv('files/validation.csv',  sep=';', quotechar='"')
    test_df  = pd.read_csv('files/test.csv',        sep=';', quotechar='"')

    le = LabelEncoder()
    y_train = le.fit_transform(train_df['account.type'])
    y_val   = le.transform(val_df['account.type'])
    y_test  = le.transform(test_df['account.type'])

    train_ds = TweetDataset(train_df['text'].tolist(),      y_train, tokenizer, MAX_LEN)
    val_ds   = TweetDataset(val_df['text'].tolist(),        y_val,   tokenizer, MAX_LEN)
    test_ds  = TweetDataset(test_df['text'].tolist(),       y_test,  tokenizer, MAX_LEN)

    return train_ds, val_ds, test_ds, le

# 4. Train loop with tqdm
def train_loop(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    prog = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for batch in prog:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        prog.set_postfix(loss=total_loss / (prog.n + 1))
    print()
    return total_loss / len(loader)

# 5. Eval loop
def eval_loop(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total

# === MAIN ===
tokenizer    = BertTokenizer.from_pretrained(MODEL_NAME)
train_ds, val_ds, test_ds, le = load_and_encode(tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

model = BERTClassifier(MODEL_NAME, num_classes=2).to(DEVICE)

# 6. Freeze all BERT parameters except pooler
for name, param in model.bert.named_parameters():
    # mantieni congelati tutti tranne pooler e ultimi due encoder layers
    if not (name.startswith("pooler") or
            name.startswith("encoder.layer.10") or
            name.startswith("encoder.layer.11") or
            name.startswith("classifier")):
        param.requires_grad = False

# Optimizer with increased weight decay
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=5e-2  # aumentato
)

criterion = nn.CrossEntropyLoss()

# -- Storici per curve --
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

best_acc = 0.0
print("Inizio training")
for epoch in range(EPOCHS):
    train_loss = train_loop(model, train_loader, optimizer, criterion, epoch)
    train_acc  = eval_loop(model, train_loader)
    val_acc    = eval_loop(model, val_loader)

    # registra metrics
    train_losses.append(train_loss)
    val_losses.append(None)   # loss su val opzionale
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss = {train_loss:.4f} | "
          f"Train Acc = {train_acc:.4f} | "
          f"Val Acc   = {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_bert_model_reg.pt")
        print("Best model saved!")

# ---------- Plot delle Curve ----------
import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, EPOCHS+1)

# Loss plot
plt.figure(figsize=(6,4))
plt.plot(epochs, train_losses, 'o-', label='Train Loss', color='tab:blue')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, ls=':')
plt.legend(); plt.tight_layout(); plt.show()

# Accuracy plot
plt.figure(figsize=(6,4))
plt.plot(epochs, train_accs, 'o--', label='Train Acc', color='tab:green')
plt.plot(epochs, val_accs,   's--', label='Val   Acc',   color='tab:red')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.ylim(0,1); plt.grid(True, ls=':')
plt.legend(); plt.tight_layout(); plt.show()

# === Test finale ===
model.load_state_dict(torch.load("best_bert_model_2.pt"))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch["input_ids"].to(DEVICE),
                        batch["attention_mask"].to(DEVICE))
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].tolist())

print("\n📊 Test Classification Report:\n",
      classification_report(all_labels, all_preds, target_names=le.classes_))
print("\n🧾 Confusion Matrix:\n",
      pd.DataFrame(confusion_matrix(all_labels, all_preds),
                   index=[f"true_{c}" for c in le.classes_],
                   columns=[f"pred_{c}" for c in le.classes_]))






import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Ricostruisci df_test allineato
df_test = pd.read_csv('files/test.csv', sep=';')
df_test['pred'] = all_preds
df_test['true'] = all_labels

# Stampa counts per account.type e class_type
print("Distribuzione account.type:\n", df_test['account.type'].value_counts())
print("Distribuzione class_type:\n", df_test['class_type'].value_counts(), "\n")

# Filtra i bot
bot_df = df_test[df_test['account.type'] == 'bot']

# Per ogni sottotipo
for subtype, subdf in bot_df.groupby('class_type'):
    acc = accuracy_score(subdf['true'], subdf['pred'])
    cm  = confusion_matrix(subdf['true'], subdf['pred'])
    print(f"--- {subtype} ---")
    print(f"  Examples: {len(subdf)}")
    print(f"  Accuracy: {acc:.4f}")
    print("  Confusion matrix:\n", cm, "\n")
