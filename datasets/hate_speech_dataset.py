import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class HateSpeechDataLoader:

    def load_dataset(file_path):
        try:
            # Utilizza pandas per caricare il file CSV direttamente
            df = pd.read_csv(file_path, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # Rimuovi eventuali virgolette attorno ai nomi delle colonne
            df.columns = df.columns.str.strip('",')

            #print("First few rows:")
            #print(df.head())

            # Stampa l'intestazione per esaminare i nomi delle colonne
            #print("Header:", df.columns)

            # Cerca gli indici delle colonne 'text' e 'label' in modo case-insensitive
            text_columns = [col for col in df.columns if 'text' in col.lower()]
            label_columns = [col for col in df.columns if 'label' in col.lower()]

            if not text_columns or not label_columns:
                raise ValueError("Columns 'text' and 'label' not found in the dataset.")

            # Estrai colonne 'text' e 'label' (può essere più di una colonna)
            texts = df[text_columns]
            labels = df[label_columns]

            # Concatena le colonne 'text' se ce ne sono più di una
            if len(text_columns) > 1:
                texts = texts.apply(lambda row: ' '.join(row.dropna()), axis=1)

            return texts, labels
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            raise ValueError("Error loading the dataset. Please check the CSV file format.")

    def split_dataset(texts, labels, test_size=0.2, random_state=42):
        return train_test_split(texts, labels, test_size=test_size, random_state=random_state)

    '''def tfidf_vectorize(X_train, X_test):
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf'''
