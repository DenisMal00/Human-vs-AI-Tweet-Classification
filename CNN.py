"""
Grid‑search Char‑CNN con salvataggio progressivo.
Se interrompi, rilancia e prosegue dai run mancanti.
"""
import os, csv, re, html, time
from collections import Counter
from itertools import product
import pandas as pd, numpy as np
import torch, torch.nn as nn
from spacy.ml.parser_model import best
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

param_grid = list(product(
    [16, 32, 64],      # emb_dim
    [64, 128, 256],          # num_filters
    [(3,4,5), (4,5,6)],     # kernel_sizes
    [0.3, 0.5],             # dropout_p
    [140,280],                  # max_len
    [64, 128, 256],         # batch_size
    [1e-3, 5e-4]                  # lr
))
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
MAX_EPOCH = 50
PATIENCE  = 10
LOG_FILE  = "grid_log.csv"
BEST_CKPT = "best_charcnn_grid.pt"

def clean(t): return re.sub(r"\s+", " ", html.unescape(t.lower())).strip()
def tokens(t): return list(t)
def build_vocab(texts, min_freq=1):
    cnt = Counter(); [cnt.update(tokens(clean(x))) for x in texts]
    vocab={"<PAD>":0,"<UNK>":1}
    for ch,f in cnt.items():
        if f>=min_freq: vocab[ch]=len(vocab)
    return vocab
def encode(txt,voc,max_len):
    ids=[voc.get(c,voc["<UNK>"]) for c in tokens(clean(txt))[:max_len]]
    return ids+[voc["<PAD>"]]*(max_len-len(ids))
class TweetSet(Dataset):
    def __init__(self,texts,labels,vocab,max_len):
        self.inputs=[encode(x,vocab,max_len) for x in texts]
        self.labels=torch.tensor(labels)
    def __len__(s): return len(s.labels)
    def __getitem__(s,i): return (torch.tensor(s.inputs[i]), s.labels[i])

class CharCNN(nn.Module):
    def __init__(self,vocab_sz,emb,kernel,num_filt,drop):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz,emb,padding_idx=0)
        self.convs=nn.ModuleList([nn.Conv1d(emb,num_filt,k) for k in kernel])
        self.drop=nn.Dropout(drop)
        self.fc  = nn.Linear(num_filt*len(kernel),2)
    def forward(self,x):
        x=self.emb(x).permute(0,2,1)
        x=[torch.relu(c(x)) for c in self.convs]
        x=[torch.max(c,2).values for c in x]
        return self.fc(self.drop(torch.cat(x,1)))

train=pd.read_csv("files/train.csv",sep=";")
val  =pd.read_csv("files/validation.csv",sep=";")
test =pd.read_csv("files/test.csv",sep=";")
for df in (train,val,test): df["clean"]=df["text"].astype(str).apply(clean)
le=LabelEncoder()
ytr=le.fit_transform(train["account.type"])
yva=le.transform(val["account.type"])
yte=le.transform(test["account.type"])
vocab=build_vocab(train["clean"])
print("Vocab size:", len(vocab))


done_runs=set()
best_val=0.0
if os.path.isfile(LOG_FILE):
    with open(LOG_FILE) as f:
        reader=csv.DictReader(f)
        for row in reader:
            done_runs.add(tuple(eval(row["params"])))
            if float(row["val_acc"])>best_val:
                best_val=float(row["val_acc"])
    print(f"Ripristinato log: {len(done_runs)} run già completate. "
          f"Best val_acc finora={best_val:.4f}")
else:
    with open(LOG_FILE,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["params","val_acc"])
        writer.writeheader()


for comb in param_grid:
    if comb in done_runs: continue   # skip
    emb,num_filt,kernel,drop,max_len,bs,lr = comb
    print(f"\n▶  Run {len(done_runs)+1}/{len(param_grid)} | {comb}")

    tr_ds=TweetSet(train["clean"],ytr,vocab,max_len)
    va_ds=TweetSet(val["clean"],  yva,vocab,max_len)

    tr_ld=DataLoader(tr_ds,batch_size=bs,shuffle=True)
    va_ld=DataLoader(va_ds,batch_size=bs)

    model=CharCNN(len(vocab),emb,kernel,num_filt,drop).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    crit=nn.CrossEntropyLoss()

    best_run,wait=0,0
    for ep in range(MAX_EPOCH):
        model.train()
        for ids,lbl in tr_ld:
            ids,lbl=ids.to(DEVICE),lbl.to(DEVICE)
            opt.zero_grad(); loss=crit(model(ids),lbl)
            loss.backward(); opt.step()

        model.eval(); corr=tot=0
        with torch.no_grad():
            for ids,lbl in va_ld:
                ids,lbl=ids.to(DEVICE),lbl.to(DEVICE)
                corr+=(model(ids).argmax(1)==lbl).sum().item(); tot+=lbl.size(0)
        val_acc=corr/tot
        if val_acc>best_run:
            best_run,wait=val_acc,0
            if val_acc>best_val:
                best_val=val_acc
                torch.save(model.state_dict(),BEST_CKPT)
        else:
            wait+=1
            if wait>=PATIENCE: break

    with open(LOG_FILE,"a",newline="") as f:
        csv.DictWriter(f,fieldnames=["params","val_acc"]).writerow(
            {"params":repr(comb),"val_acc":f"{best_run:.4f}"})
    done_runs.add(comb)
    print(f"   best_val_acc_run={best_run:.4f} | global_best={best_val:.4f}")

best_emb,best_filt,best_k,best_drop,best_len,best_bs,best_lr = best(
    [eval(r["params"]) for r in csv.DictReader(open(LOG_FILE))],
    key=lambda p: float([r["val_acc"] for r in csv.DictReader(open(LOG_FILE))
                         if r["params"]==repr(p)][0]))
print("\n🏁 Grid search terminata.")
print("Migliori parametri:", best_emb,best_filt,best_k,best_drop,best_len,best_bs,best_lr)
print("Miglior val_acc:", best_val)

te_ds = TweetSet(test["clean"], yte, vocab, best_len)
te_ld = DataLoader(te_ds, batch_size=best_bs)
model = CharCNN(len(vocab), best_emb, best_k, best_filt, best_drop).to(DEVICE)
model.load_state_dict(torch.load(BEST_CKPT))
model.eval(); preds, labs=[],[]
with torch.no_grad():
    for ids,lbl in te_ld:
        preds.extend(model(ids.to(DEVICE)).argmax(1).cpu().tolist())
        labs.extend(lbl.tolist())
print("\n📊 Test report:")
print(classification_report(labs, preds, target_names=le.classes_))
print("\n🧾 Confusion matrix:")
print(pd.DataFrame(
        confusion_matrix(labs, preds),
        index=[f"true_{c}" for c in le.classes_],
        columns=[f"pred_{c}" for c in le.classes_]
))

