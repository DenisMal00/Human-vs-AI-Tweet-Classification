import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd

MODEL_NAME = "bert-base-uncased"
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 0
LR         = 2e-5
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

#Load and preprocess data
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

# MAIN
tokenizer    = BertTokenizer.from_pretrained(MODEL_NAME)
train_ds, val_ds, test_ds, le = load_and_encode(tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

model = BERTClassifier(MODEL_NAME, num_classes=2).to(DEVICE)

# Freeze all BERT parameters except pooler
for name, param in model.bert.named_parameters():
    if not (name.startswith("pooler") or
            name.startswith("encoder.layer.10") or
            name.startswith("encoder.layer.11") or
            name.startswith("classifier")):
        param.requires_grad = False

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=5e-2
)

criterion = nn.CrossEntropyLoss()

# Final Test
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

