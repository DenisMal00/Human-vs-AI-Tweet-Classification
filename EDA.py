import re
import pandas as pd
import html
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer



# Leggi il file CSV
train_df = pd.read_csv('files/train.csv', sep=';', quotechar='"', engine='python')
val_df = pd.read_csv('files/validation.csv', sep=';', quotechar='"', engine='python')
test_df = pd.read_csv('files/test.csv', sep=';', quotechar='"', engine='python')

train_df_clean = pd.read_csv('files/train_clean.csv', sep=';', quotechar='"', engine='python')
val_df_clean = pd.read_csv('files/validation_clean.csv', sep=';', quotechar='"', engine='python')
test_df_clean = pd.read_csv('files/test_clean.csv', sep=';', quotechar='"', engine='python')

pd.set_option('display.max_colwidth', None)  # Mostra il contenuto completo delle colonne testuali
#print(train_df_clean.iloc[0].to_string())

df_uncleaned = pd.concat([train_df, val_df, test_df], ignore_index=True)
df = pd.concat([train_df_clean, val_df_clean, test_df_clean], ignore_index=True)


# Distribuzione classi
print("\n📊 Distribuzione account.type:")
print(df['account.type'].value_counts())

print("\n📊 Distribuzione class_type:")
print(df['class_type'].value_counts())

# Lunghezza testi
df['text_clean']=df['text_clean'].fillna("")
df['num_words'] = df['text_clean'].apply(lambda x: len(x.split()))
df['num_chars'] = df['text_clean'].apply(len)

df_uncleaned['num_words'] = df_uncleaned['text'].apply(lambda x: len(x.split()))
df_uncleaned['num_chars'] = df_uncleaned['text'].apply(len)

print("\n📊 Media parole/caratteri per tipo di account:")
print(df.groupby('account.type')[['num_words', 'num_chars']].mean())
print(df_uncleaned.groupby('account.type')[['num_words', 'num_chars']].mean())


plt.figure(figsize=(10, 4))
sns.boxplot(x='account.type', y='num_words', data=df)
plt.title("Numero di parole per tipo di account")
#plt.show()

# Diversità lessicale
def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if words else 0

df['lexical_div'] = df['text_clean'].apply(lexical_diversity)

plt.figure(figsize=(10, 4))
sns.violinplot(x='account.type', y='lexical_div', data=df)
plt.title("Diversità lessicale per tipo di account")
#plt.show()

# Parole più comuni
def get_top_words(texts):
    vectorizer = CountVectorizer(stop_words='english', max_features=15)
    word_matrix = vectorizer.fit_transform(texts)
    word_freq = word_matrix.sum(axis=0)
    return pd.DataFrame({
        'word': vectorizer.get_feature_names_out(),
        'frequency': word_freq.tolist()[0]
    }).sort_values(by='frequency', ascending=False)

top_words_human = get_top_words(df[df['account.type'] == 'human']['text_clean'])
top_words_bot = get_top_words(df[df['account.type'] == 'bot']['text_clean'])
print(top_words_human)
print(top_words_bot)
plt.figure(figsize=(10, 4))
sns.barplot(data=top_words_human, x='frequency', y='word')
plt.title("Top parole - Human")
#plt.show()

plt.figure(figsize=(10, 4))
sns.barplot(data=top_words_bot, x='frequency', y='word')
plt.title("Top parole - Bot")
#plt.show()