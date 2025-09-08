import html

import numpy as np
import pandas as pd


import re
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('files/train.csv', sep=';', quotechar='"', engine='python')
val_df = pd.read_csv('files/validation.csv', sep=';', quotechar='"', engine='python')
test_df = pd.read_csv('files/test.csv', sep=';', quotechar='"', engine='python')
def clean_text_base(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text.lower())
    text = re.sub(r"https?://\\S+", "__url__", text)
    text = re.sub(r"@\\w+", "__user_mention__", text)
    text = re.sub(r"#", "", text)  # rimuovi solo '#'
    return re.sub(r"\\s+", " ", text).strip()

for df in [train_df, val_df, test_df]:
    df['text_clean'] = df['text'].apply(clean_text_base)

def count_hashtags(text):
    return len(re.findall(r"#\w+", text))

def count_mentions(text):
    return len(re.findall(r"@\w+", text))

def count_links(text):
    return len(re.findall(r"http\S+", text))

def add_structure_features(df):
    df['num_hashtags'] = df['text'].apply(count_hashtags)
    df['num_mentions'] = df['text'].apply(count_mentions)
    df['num_links'] = df['text'].apply(count_links)
    return df

# Unisci i dataset se vuoi analisi globale
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Aggiungi le nuove colonne
full_df = add_structure_features(full_df)

# Raggruppa per tipo (bot vs human, usando 'account.type')
grouped = full_df.groupby('account.type')[['num_hashtags', 'num_mentions', 'num_links']].mean()
print("\n📊 Media elementi strutturali per tipo di account:\n")
print(grouped)


import pandas as pd
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt

def count_emojis(text):
    emoji_pattern = re.compile(r'<U\+.*?>', flags=re.UNICODE)
    return len(emoji_pattern.findall(text))

def count_uppercase_words(text):
    return len(re.findall(r"\b[A-Z]{2,}\b", text))

def count_punctuation(text):
    return sum(1 for char in text if char in string.punctuation)

def add_extra_structural_features(df):
    df['num_emojis'] = df['text'].apply(count_emojis)
    df['num_uppercase'] = df['text'].apply(count_uppercase_words)
    df['num_punct'] = df['text'].apply(count_punctuation)
    return df

# Unione dei dataset se non l'hai già fatto
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Aggiunta colonne strutturali extra
full_df = add_extra_structural_features(full_df)

# Media per tipo di account
grouped_extra = full_df.groupby('account.type')[['num_emojis', 'num_uppercase', 'num_punct']].mean()
print("\n📊 Media emoji / maiuscole / punteggiatura per tipo di account:\n")
print(grouped_extra)


# Unione dei dataset
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Funzioni di analisi testuale
def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0

def has_repetitions(text):
    return int(bool(re.search(r'\b(\w+)( \1)+\b', text.lower())))

def starts_with(text):
    return text.split()[0].lower() if isinstance(text, str) and len(text.split()) > 0 else ""

'''
def starts_with_two(text):
    words = text.split()
    return " ".join(words[:2]).lower() if len(words) >= 2 else (words[0].lower() if words else "")
'''

# Aggiunta colonne
full_df['num_words'] = full_df['text'].apply(lambda x: len(x.split()))
full_df['num_chars'] = full_df['text'].apply(len)
full_df['lexical_div'] = full_df['text'].apply(lexical_diversity)
full_df['has_repeats'] = full_df['text'].apply(has_repetitions)
full_df['starts_with'] = full_df['text'].apply(starts_with)

# Confronto tra account.type (human vs bot)
comparison = full_df.groupby('account.type').agg({
    'num_words': 'mean',
    'num_chars': 'mean',
    'lexical_div': 'mean',
    'has_repeats': 'mean',
    'starts_with': lambda x: x.value_counts().idxmax()
}).rename(columns={
    'num_words': 'Media parole',
    'num_chars': 'Media caratteri',
    'lexical_div': 'Diversità lessicale media',
    'has_repeats': 'Frequenza ripetizioni',
    'starts_with': 'Parola iniziale più comune'
})

# Mostra risultato
print("\n📊 Confronto tra testi umani e bot:")
from tabulate import tabulate
print(tabulate(comparison, headers='keys', tablefmt='pretty'))







import numpy as np
import matplotlib.pyplot as plt

# lunghezze parole (già calcolate)
wc_h = full_df.loc[full_df['account.type']=='human', 'num_words'].clip(upper=60)
wc_b = full_df.loc[full_df['account.type']=='bot',   'num_words'].clip(upper=60)

bins = np.arange(0, 61, 2)

plt.figure(figsize=(6,4))
plt.hist(
    wc_h, bins=bins,
    weights=np.ones_like(wc_h) * 100 / len(wc_h),   # ogni barra somma al 100%
    alpha=0.8, label='Human', color='#853861', edgecolor='none'
)
plt.hist(
    wc_b, bins=bins,
    weights=np.ones_like(wc_b) * 100 / len(wc_b),
    alpha=0.7, label='Bot',   color='#4C5156', edgecolor='none'
)

plt.xlabel('Words per tweet (clipped at 60)')
plt.ylabel('Percentage of tweets (%)')
plt.title('Tweet length distribution (words)')
plt.legend(frameon=False)
plt.tight_layout()
#plt.show()



# ===================== % TWEET CON ≥1 OCCORRENZA =====================

def pct_with(df, col):
    return (df[col] > 0).mean() * 100

human = full_df.query("`account.type`=='human'")
bot   = full_df.query("`account.type`=='bot'")

features_bin = {
    'num_mentions':  'Mentions',
    'num_links':     'Links',
    'num_hashtags':  'Hashtags',
    'num_emojis':    'Emojis',
    'num_uppercase': 'ALL-CAPS words',
    'num_punct':     'Punctuation marks ≥1'  # opzionale, spesso è sempre >0
}

rows = []
for col, nice in features_bin.items():
    h = pct_with(human, col)
    b = pct_with(bot, col)
    rows.append([nice, round(h,1), round(b,1), f"{h-b:+.1f}"])

table_pct = pd.DataFrame(rows, columns=["Feature", "Human %", "Bot %", "Δ (pp)"])
print("\n=== % of tweets with ≥1 occurrence ===")
print(table_pct.to_string(index=False))

# (Opzionale) “1 ogni X tweet” per leggere la media più intuitivamente:
def one_every(mean_val):
    return f"1 every ~{(1/mean_val):.1f} tweets" if mean_val>0 else "—"

mean_rows = []
for col, nice in [('num_mentions','Mentions'),
                  ('num_links','Links'),
                  ('num_emojis','Emojis'),
                  ('num_uppercase','ALL-CAPS words')]:
    mh = human[col].mean()
    mb = bot[col].mean()
    mean_rows.append([nice,
                      one_every(mh),
                      one_every(mb)])
table_every = pd.DataFrame(mean_rows, columns=["Feature", "Human (avg)", "Bot (avg)"])
print("\n=== Readable: how often it appears on average ===")
print(table_every.to_string(index=False))
