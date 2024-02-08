import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.utils import resample

nltk.download('punkt')

def clean_data(dataset):
    ## Verranno eseguiti vari step di pulizia per dati testuali

    # Rimozione di "\r" e "\n"
    dataset["comment_text"] = dataset["comment_text"].apply(lambda x: re.sub(r'[\r\n]+', '', x))
    # Rimozione di sequenze di ":" (esempio, "::::")
    dataset["comment_text"] = dataset["comment_text"].apply(lambda x: re.sub(r'::+', '', x))
    # Rimozione di sequenze di "=" (esempio, "====")
    dataset["comment_text"] = dataset["comment_text"].apply(lambda x: re.sub(r'==+', '', x))
    # Rimozione di sequenze di "*" (esempio, "**")
    dataset["comment_text"] = dataset["comment_text"].apply(lambda x: re.sub(r'\*\*+', '', x))
    # Rimozione di sequenze numeriche in formato di indirizzi IP (esempio, "192.168.1.1")
    dataset["comment_text"] = dataset["comment_text"].apply(lambda x: re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '', x))
    # Rimozione di contenuto compreso tra Parentesi Quadre (esempio, "[contentContent]")
    dataset["comment_text"] = dataset["comment_text"].apply(lambda x: re.sub(r'\[[^\[\]]+\]', '', x))
    # Rimozione di Apici, sia singoli che doppi
    dataset["comment_text"] = dataset["comment_text"].apply(lambda x: re.sub(r"['\"]", "", x))

    ## La rimozione di particolari caratteri o sequenze di caratteri può portare alla fusione di due token diversi

    # Splitting di token in cui compare un segno di interpuzione forte ("?", "!" e ".") seguito da una lettera maiuscola
    dataset["comment_text"] = dataset["comment_text"].apply(lambda x: re.sub(r'([?!\.])([A-Z]\w*)', r'\1 \2', x))
    # Splitting di parole fuse (esempio, "parolaParola" diventa "parola Parola")
    dataset["comment_text"] = dataset["comment_text"].apply(lambda x: re.sub(r'([a-z])([A-Z])', r'\1 \2', x))

    return dataset

def transform_data(dataset):
    ## Trasformazione di tutte le lettere maiuscole in minuscole e rimozione di tutti i segni di interpunzione

    phrases = dataset["comment_text"].to_list()
    phrases_cleaned = list()

    for phrase in phrases:
        tokens = word_tokenize(phrase)
        lowercase_tokens = [token.lower() for token in tokens if token.isalpha()]
        phrases_cleaned.append(' '.join(lowercase_tokens))

    dataset["comment_text"] = pd.Series(phrases_cleaned)
    
    return dataset

def build_training_set(dataset):
    toxic_entries = dataset[dataset['toxic'] == 1]
    non_toxic_entries = dataset[dataset['toxic'] == 0]
    print("Numero di Frasi 'toxic': " + str(len(toxic_entries)) + ", Numero di Frasi 'non-toxic': " + str(len(non_toxic_entries)))

    non_toxic_downsampled = resample(non_toxic_entries, n_samples=len(toxic_entries), random_state=42)

    final_training_set = pd.concat([toxic_entries, non_toxic_downsampled])
    final_training_set.reset_index(drop=True, inplace=True)

    print(final_training_set)

    idx_to_remove = list()
    for i in range(0, len(final_training_set)):
        row = final_training_set.iloc[i]
        if row['comment_text'] is '':
            idx_to_remove.append(i)
  
    final_training_set = final_training_set.drop(idx_to_remove)
    final_training_set = final_training_set[['comment_text', 'toxic']]

    return final_training_set

### MAIN ###
train_data = pd.read_csv("./jigsaw_train_set.csv")

if train_data is not None:
    print("Training Set RAW caricato con successo!")
    print("Inizio del Preprocessing...")
    train_data = clean_data(train_data)
    train_data = transform_data(train_data)
    
    print("\nCompletata la pulizia del testo, creazione del Training Set finale.")
    print("Creiamo un Dataset bilanciato tra esempi 'toxic' e 'non_toxic'.\n")
    train_data = build_training_set(train_data)

    train_data.to_csv("./training_set.csv", index=False)
    print("Fine del Preprocessing! Memorizzazione del Training Set elaborato effettuata, dimensione = " + str(len(train_data)))