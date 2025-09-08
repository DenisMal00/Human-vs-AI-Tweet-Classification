from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.sparse import hstack
import re
import pandas as pd
import html
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

import string

def count_punctuation(text):
    return sum(1 for char in text if char in string.punctuation)

def repetition_ratio(text):
    words = text.lower().split()
    if not words:
        return 0.0
    total = len(words)
    unique = len(set(words))
    return 1 - (unique / total)

def extract_features(df, text_col='text'):
    df = df.copy()
    df['has_url'] = df[text_col].str.contains(r'http\S+', regex=True).astype(int)
    df['has_mention'] = df[text_col].str.contains(r'@\w+', regex=True).astype(int)
    df['has_hashtag'] = df[text_col].str.contains(r'#\w+', regex=True).astype(int)
    df['num_punctuations'] = df[text_col].apply(count_punctuation)
    df['num_uppercase_words'] = df[text_col].apply(lambda x: sum(1 for word in x.split() if word.isupper()))
    df['num_chars'] = df[text_col].str.len()
    df['num_words'] = df[text_col].apply(lambda x: len(x.split()))
    df['text_len_ratio'] = df.apply(lambda row: row['num_chars'] / row['num_words'] if row['num_words'] > 0 else 0, axis=1)
    df['repetition_ratio'] = df[text_col].apply(repetition_ratio)

    return df

train_df = pd.read_csv('files/train_clean.csv', sep=';', quotechar='"', engine='python')
val_df = pd.read_csv('files/validation_clean.csv', sep=';', quotechar='"', engine='python')
test_df = pd.read_csv('files/test_clean.csv', sep=';', quotechar='"', engine='python')

for df in [train_df, val_df, test_df]:
    df['text'] = df['text'].fillna("")


def remove_stopwords_sklearn(text):
    tokens = text.split()
    return " ".join([word for word in tokens if word.lower() not in ENGLISH_STOP_WORDS])

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])

MENTION_RE = re.compile(r"@\w{1,15}")
URL_RE = re.compile(
    r"""(?xi)         
    \b
    (?:https?://|www\.)           
    [\w._~:/?#\[\]@!$&'()*+,;=%-]+ 
    """
)
def clean_text_base(text: str) -> str:
    """Lower‑case, unescape HTML, sostituisci URL e menzioni."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r'&?gt;?', '>', text)  # &gt , &gt; , gt -> >
    text = re.sub(r'&amp;gt;?', '>', text)  # &amp;gt , &amp;gt; -> >
    text = re.sub(r'\bsrc\b=?', '__src_tag__', text)

    text = html.unescape(text.lower())

    text = URL_RE.sub("__url__", text)


    text = MENTION_RE.sub("__user_mention__", text)

    text = text.replace("#", "")

    text = re.sub(r"\\s+", " ", text).strip()

    return text

for df in [train_df, val_df, test_df]:
    df['text_clean'] = df['text'].apply(clean_text_base)



train_df = extract_features(train_df)
val_df = extract_features(val_df)
test_df = extract_features(test_df)

vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 3),sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(train_df['text_clean'])
X_val_tfidf = vectorizer.transform(val_df['text_clean'])
X_test_tfidf = vectorizer.transform(test_df['text_clean'])

extra_features = [ 'has_mention','has_url', 'has_hashtag']
X_train_extra = train_df[extra_features].values
X_val_extra = val_df[extra_features].values
X_test_extra = test_df[extra_features].values

X_train_final = hstack([X_train_tfidf, X_train_extra])
X_val_final = hstack([X_val_tfidf, X_val_extra])
X_test_final = hstack([X_test_tfidf, X_test_extra])

y_train = train_df['account.type']
y_val = val_df['account.type']
y_test = test_df['account.type']

clf = LinearSVC(C=1)
clf.fit(X_train_final, y_train)

y_val_pred = clf.predict(X_val_final)
print("\n📊 Classification Report (Validation):\n")
print(classification_report(y_val, y_val_pred))
print("📉 Confusion Matrix (Validation):\n")
print(confusion_matrix(y_val, y_val_pred))

y_test_pred = clf.predict(X_test_final)
print("\n🧪 Classification Report (Test):\n")
print(classification_report(y_test, y_test_pred))
print("📉 Confusion Matrix (Test):\n")
print(confusion_matrix(y_test, y_test_pred))

'''
# ------------------------------------------------------------
# TOP / BOTTOM FEATURE WEIGHTS
# ------------------------------------------------------------
import numpy as np

# 1) vettore dei pesi
coef = clf.coef_.ravel()

# 2) se 'bot' NON è la seconda etichetta, inverti il segno
#    (vogliamo: peso positivo  ->  favorisce 'bot')
if clf.classes_[1] != 'bot':
    coef = -coef                # nel tuo caso non serve, ma è generale

# 3) nomi delle feature: TF‑IDF + extra
tf_names   = vectorizer.get_feature_names_out()
feat_names = np.concatenate([tf_names, extra_features])

# 4) stampa i primi/ultimi N
N = 30
top_bot   = coef.argsort()[::-1][:N]   # pesi più positivi
top_human = coef.argsort()[:N]         # pesi più negativi

print(f"\n🔝 Top {N} feature pro‑BOT:")
for idx in top_bot:
    print(f"{feat_names[idx]:25s}  {coef[idx]:+.3f}")

print(f"\n🔝 Top {N} feature pro‑HUMAN:")
for idx in top_human:
    print(f"{feat_names[idx]:25s}  {coef[idx]:+.3f}")


import matplotlib.pyplot as plt
N = 15
top_idx = np.argsort(coef)[-N:]
bot_idx = np.argsort(coef)[:N]
plt.figure(figsize=(6,4))
plt.barh(feat_names[top_idx], coef[top_idx], color='red')
plt.title(f'Top {N} BOT‑driving features'); plt.show()

plt.figure(figsize=(6,4))
plt.barh(feat_names[bot_idx], coef[bot_idx], color='blue')
plt.title(f'Top {N} HUMAN‑driving features'); plt.show()



import numpy as np, pandas as pd, re

# -------- PARAMETRI PRINCIPALI ----------------------------------
MODE          = "false_neg"          # "bot", "human", "borderline", "false_pos", "false_neg"
NUM_EXAMPLES  = 0
THR_LO, THR_HI = 0.10, 0.50      # soglie colore
# ----------------------------------------------------------------

# -------- 1) PESI DELLE FEATURE (+ = BOT, - = HUMAN) ------------
coef = clf.coef_.ravel().copy()
if clf.classes_[1] != 'bot':      # flip se necessario
    coef *= -1

feat_names = vectorizer.get_feature_names_out()
feat2w     = {feat_names[i]: coef[i] for i in range(len(feat_names))}

# -------- 2) PREPROCESS & TOKENIZER DEL VECTORIZER --------------
preproc   = vectorizer.build_preprocessor()
tokenizer = vectorizer.build_tokenizer()

def tokens_unigram(txt):
    """token list coerente con il modello (solo unigram)."""
    return tokenizer(preproc(txt))

# -------- 3) AGGREGAZIONE PESI SU n‑GRAM 1..3 -------------------
def aggregate_weights(tokens, n_max=3):
    n   = len(tokens)
    agg = [0.0]*n
    for k in range(1, n_max+1):
        for i in range(n-k+1):
            key = " ".join(tokens[i:i+k])
            w   = feat2w.get(key, 0.0)
            if w:
                for j in range(i, i+k):
                    agg[j] += w
    return agg

# -------- 4) COLORAZIONE ----------------------------------------
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def highlight_line(txt):
    toks = tokens_unigram(txt)
    weights = aggregate_weights(toks, n_max=3)
    out=[]
    for tok, w in zip(toks, weights):

        # -- NEW: salta stop‑word o token corti  (opzionale) ------------
        if tok in ENGLISH_STOP_WORDS or len(tok) < 0:
            out.append(tok); continue
        # ----------------------------------------------------------------

        if   w >  THR_HI: col = 41
        elif w > THR_LO:  col = 101
        elif w < -THR_HI: col = 44
        elif w < -THR_LO: col = 104
        else:             out.append(tok); continue
        out.append(f"\x1b[{col}m{tok}\x1b[0m")
    return " ".join(out)

# -------- 5) SCORE E PREVISIONI COERENTI ------------------------
scores = clf.decision_function(X_test_final)
if clf.classes_[1] != 'bot':      # flip lo score come i pesi
    scores *= -1

df = test_df.copy()
df['score'] = scores
df['pred']  = np.where(scores > 0, 'bot', 'human')
df['true']  = y_test

# -------- 6) SELEZIONE MODALITÀ ---------------------------------
if MODE == "bot":
    subset = df[df['pred']=='bot'].sort_values('score', ascending=False)
elif MODE == "human":
    subset = df[df['pred']=='human'].sort_values('score')
elif MODE == "borderline":
    subset = df.iloc[(df['score'].abs()).argsort()]
elif MODE == "false_pos":
    subset = df[(df['pred']=='bot') & (df['true']=='human')].sort_values('score', ascending=False)
elif MODE == "false_neg":
    subset = df[(df['pred']=='human') & (df['true']=='bot')].sort_values('score')
else:
    raise ValueError("MODE non valido")

# -------- 7) STAMPA TWEET E COLORI ------------------------------
# -------- 7) STAMPA TWEET E COLORI ------------------------------
for idx, row in subset.head(NUM_EXAMPLES).iterrows():
    # idx è l'indice del DataFrame (quello che vedi in df.iloc[...] / df.loc[idx])
    print(f"\n┌─ ID {idx} | SCORE {row.score:+.3f} | pred={row.pred} | true={row.true}")
    print("│ " + highlight_line(row.text_clean))
    print("└" + "─"*80)


def print_colored_tweet(tweet_id):
    """
    Stampa il singolo tweet identificato da tweet_id
    con le parole evidenziate in base ai pesi.
    """
    # Recupera la riga corrispondente
    try:
        row = df.loc[tweet_id]
    except KeyError:
        print(f"ID {tweet_id} non trovato nel DataFrame.")
        return

    # Stampa header
    score = row['score']
    pred = row['pred']
    true = row['true']
    print(f"\n┌─ ID {tweet_id} | SCORE {score:+.3f} | pred={pred} | true={true}")

    # Evidenzia e stampa il testo
    colored = highlight_line(row['text_clean'])
    print("│ " + colored)
    print("└" + "─" * 80)


# Esempio d’uso:
print_colored_tweet(752)


# ======================================================================
#  ANALISI DI GENERALIZZAZIONE – learning‑curve  +  test benchmark
# ======================================================================
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import learning_curve, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

print("\n… costruisco learning‑curve SVM (word n‑gram) …")

# ------------------------------------------------------------------
# 1)   usa SOLO train+validation per la curva                     --
#      (lo chiamiamo X_dev / y_dev)                               --
# ------------------------------------------------------------------
import scipy.sparse as sp
X_dev = sp.vstack([X_train_final, X_val_final])     # sparse vstack
y_dev = pd.concat([y_train, y_val], ignore_index=True)

base_svm = LinearSVC(C=1, loss="hinge")
cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
        estimator   = base_svm,
        X           = X_dev,
        y           = y_dev,
        train_sizes = np.linspace(0.1, 1.0, 8),
        cv          = cv,
        scoring     = "accuracy",
        n_jobs      = -1,
        shuffle     = True,
        random_state= 42
)

val_mean, val_std = val_scores.mean(1), val_scores.std(1)

# ------------------------------------------------------------------
#import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.svm import LinearSVC
from scipy.sparse import vstack
from sklearn.metrics import accuracy_score

# 1) Prepara X_dev = train+val per il CV interno
X_dev = vstack([X_train_final, X_val_final]).tocsr()
y_dev = pd.concat([y_train, y_val], ignore_index=True)

# 2) Learning curve (accuracy CV)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
est = LinearSVC(C=0.316, loss="hinge", max_iter=10000, random_state=42)

train_sizes, _, val_scores = learning_curve(
    estimator=est,
    X=X_dev, y=y_dev,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=cv, scoring="accuracy",
    n_jobs=-1, shuffle=True, random_state=42
)

val_mean = val_scores.mean(axis=1)

# 3) Test accuracy
test_acc = accuracy_score(y_test, clf.predict(X_test_final))

# 4) Plot senza fill_between
plt.figure(figsize=(6, 4))
plt.plot(train_sizes, val_mean, "s-", color="tab:orange", label="CV accuracy (5‑fold)")
plt.axhline(test_acc, ls="--", lw=2, color="tab:green", label=f"Test accuracy = {test_acc:.3f}")

plt.ylim(min(val_mean) - .02, max(val_mean) + .04)
plt.xlabel("Numero di esempi di training")
plt.ylabel("Accuracy")
plt.title("Learning‑curve SVM (word n‑gram)")
plt.legend(loc="lower right")
plt.grid(True, ls=":")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 4)   Accuracy singole fold (utile varianza per slide)           --
# ------------------------------------------------------------------
fold_acc = cross_val_score(base_svm, X_dev, y_dev,
                           cv=cv, scoring="accuracy", n_jobs=-1)
print("\nFold‑accuracy 5‑CV:", ", ".join(f"{a:.3f}" for a in fold_acc))
print(f"Media ± std : {fold_acc.mean():.4f}  ± {fold_acc.std():.4f}")
print(f"Accuracy TEST: {test_acc:.4f}")

# ------------------------------------------------------------------
# 5)   Confusion‑matrix sul TEST                                  --
# ------------------------------------------------------------------
cm = confusion_matrix(y_test, y_test_pred, labels=['bot', 'human'])
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['bot', 'human'],
            yticklabels=['bot', 'human'])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion matrix – test")
plt.tight_layout(); plt.show()

# ------------------------------------------------------------------
# 6)   Qualche esempio di errori                                  --
# ------------------------------------------------------------------
df_test = test_df.copy()
df_test["pred"] = y_test_pred
# falsi positivi: pred 'bot' ma era 'human'
fp = df_test[(df_test.pred == "bot") & (df_test["account.type"] == "human")] \
        .head(5)["text_clean"].tolist()
# falsi negativi: pred 'human' ma era 'bot'
fn = df_test[(df_test.pred == "human") & (df_test["account.type"] == "bot")] \
        .head(5)["text_clean"].tolist()

print("\n≈≈≈  FALSE POSITIVES (sample)  ≈≈≈")
for t in fp: print("•", t[:120], "…")

print("\n≈≈≈  FALSE NEGATIVES (sample)  ≈≈≈")
for t in fn: print("•", t[:120], "…")

'''