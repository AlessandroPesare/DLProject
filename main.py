import numpy as np
from features.feature_extractor import FeatureExtractor
from trainers.classifier_trainer import ClassifierTrainer
from datasets.hate_speech_dataset import HateSpeechDataLoader
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    # Path verso il dataset
    dataset_path = '/Users/alessandropesare/v0.2.2.csv'

    # Carica il dataset
    texts, labels = HateSpeechDataLoader.load_dataset(dataset_path)

    # Dividi il dataset
    X_train, X_test, y_train, y_test = HateSpeechDataLoader.split_dataset(texts, labels)

    # Codifica delle etichette
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    print(y_train_encoded[:5])
    # Inizializza l'estrattore di caratteristiche
    feature_extractor = FeatureExtractor()
    # Stampa di debug per visualizzare le caratteristiche BERT per un singolo testo
    sample_text = X_train.iloc[0]['text']
    bert_features = feature_extractor.extract_features_bert(sample_text)
    print("BERT features shape for a single text:", bert_features.shape)
    # Estrazione delle caratteristiche con BERT
    X_train_bert = [feature_extractor.extract_features_bert(text) for text in X_train]
    X_test_bert = [feature_extractor.extract_features_bert(text) for text in X_test]

    # Converti le liste in array numpy
    X_train_bert = np.array(X_train_bert)  
    X_test_bert = np.array(X_test_bert)

    print("Shape of X_train_bert:", X_train_bert.shape)
    print("Shape of X_test_bert:", X_test_bert.shape)
    print("Length of X_train:", len(X_train))
    print("Length of X_test:", len(X_test))


    # Addestramento e valutazione del classificatore con BERT
    svm_classifier_with_bert = SVC()
    classifier_trainer_with_bert = ClassifierTrainer(svm_classifier_with_bert)

    classifier_trainer_with_bert.train(X_train_bert, y_train_encoded.ravel())
    accuracy_with_bert = classifier_trainer_with_bert.evaluate(X_test_bert, y_test_encoded.to_numpy().ravel())
    print(f'Accuracy with BERT: {accuracy_with_bert}')

    # Estrazione delle caratteristiche con TF-IDF
    X_train_tfidf, X_test_tfidf = feature_extractor.extract_features_tfidf(X_train, X_test)

    print("Shape of X_train_tfidf:", X_train_tfidf.shape)
    print("Shape of X_test_tfidf:", X_test_tfidf.shape)

    # Addestramento e valutazione del classificatore con TF-IDF
    svm_classifier_with_tfidf = SVC()
    classifier_trainer_with_tfidf = ClassifierTrainer(svm_classifier_with_tfidf)

    classifier_trainer_with_tfidf.train(X_train_tfidf, y_train)
    accuracy_with_tfidf = classifier_trainer_with_tfidf.evaluate(X_test_tfidf)
    print(f'Accuracy with TF-IDF: {accuracy_with_tfidf}')
