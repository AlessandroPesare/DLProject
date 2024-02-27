# Exploring BERT for Hate Detection

Il presente Repository contiene il Codice ed i Notebook relativi al Progetto "Exploring BERT for Hate Detection", sviluppato da [Alessandro Pesare](https://www.github.com/AlessandroPesare) e [Riccardo De Cesaris](https://www.github.com/DecioXXIV) nell'ambito del Corso di Deep Learning dell'anno accademico 2023-24.

L'obbiettivo di tale progetto è di carattere esplorativo: vengono messi a paragone i risultati ottenuti da determinati Classificatori nell'ambito del Task di "Hate Detection" con e senza l'impiego di BERT come Feature Extractor per le istanze testuali. Di seguito si descrive la struttura del repository:

- download_files.py: script per scaricare localmente tutti i Dataset utilizzati.
- utils: Directory che contiene degli script utili per il pre-processing dei Dataset e per l'estrazione delle Feature con BERT. I Dataset già elaborati sono comunque disponibili eseguendo lo script "download_files.py".
- models: Directory contenente i Notebook descrittivi dell'esecuzione dei Classificatori.
  - base: Random Forest 500, Random Forest 1000, Linear SVM, NN Small e NN Big supportate dalla Pipeline "non-Lemmatizzata" (TFIDF -> Classificatore) e "Lemmatizzata" (Lemmatizzazione -> TFIDF -> Classificatore).
  - bert_boosted: Random Forest 500, Random Forest 1000, Linear SVM, NN Small e NN Big supportate dalla Pipeline "BERT-based" (BERT -> Classificatore).
- explanations: Directory contenente i Notebook descrittivi delle Explanation (generate con LIME) per le classificazioni. I Notebook sono nuovamente suddivisi in "base" e "bert_boosted".
