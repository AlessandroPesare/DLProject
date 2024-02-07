import gdown
import zipfile
import os

def download_datasets():
    print("DATASET DOWNLOADING")
    dataset_folder = "./datasets/"
    file_dict = {
        'jigsaw_train_set.csv': 'https://drive.google.com/uc?export=download&id=1DG3WQxA-Qx358k1bcGXtmkdwXMMGM4yF',
        'jigsaw_test_set.csv': 'https://drive.google.com/uc?export=download&id=14QHA6-99fIGqhQm-occTMvk00JNGhqnA',
        'training_set.csv': 'https://drive.google.com/uc?export=download&id=1oK-jSYp26iAlpM0otPEapCEfgdmfb7Fv',
        'training_set_lemmatized.csv': 'https://drive.google.com/uc?export=download&id=1nJvsWvwk_2lWtlmsTEp2i_-coQwI0OFN',
        'test_set.csv': 'https://drive.google.com/uc?export=download&id=112Q6ebP109U-bvPNcRhSUTL20-_uGKf4',
        'test_set_lemmatized.csv': 'https://drive.google.com/uc?export=download&id=1rJ2kGp7J3E-P7O22_eshpPO0ZloJbx_Q',
        'X_train_bert.csv': 'https://drive.google.com/uc?export=download&id=1Ighh847Te0cG8OFqVVaThNp7Yu9i-HCe',
        'X_test_bert.csv': 'https://drive.google.com/uc?export=download&id=13qt1iTT9v6i2SdlpY3HRp5W9eZCpd46R'
        }
    
    for entry in file_dict.items():
        url = entry[1]
        file_name = dataset_folder + entry[0]
        if os.path.exists(file_name) is False:
            print("Downloading File:", entry[0])
            gdown.download(url=url, output=file_name)
        else:
            print("File '" + str(entry[0]) + "' already exists")
    
    print("Dataset Downloaded.\n\n")

def download_pretrained_base_models():
    print("BASE MODELS DOWNLOADING")
    base_model_folder = "./models/base/"
    models_dict = {
        'rf_classifier_100.zip': 'https://drive.google.com/uc?export=download&id=1gosZgq3LQkcDf0HgpTyC1ERwyiG6smJW',
        'rf_classifier_lem_100.zip': 'https://drive.google.com/uc?export=download&id=1JyQtwkgZ40CwJHR3B3Wq3InGIgZl7DdC',
        'rf_classifier_500.zip': 'https://drive.google.com/uc?export=download&id=14DuDqx15NLGLM_H4Q4qEMb4bxH-G9uLz',
        'rf_classifier_lem_500.zip': 'https://drive.google.com/uc?export=download&id=1ZIX86p6GgHY-VNOACWT6F_u0E7kqWNp0',
        'linear_svm_classifier.zip': 'https://drive.google.com/uc?export=download&id=1KNZ3B2ebSy6LRUJMGOdvQXM0Pa_917Dv',
        'linear_svm_classifier_lem.zip': 'https://drive.google.com/uc?export=download&id=1ac_YllwyMC5jciY1mrn15oKEEdvctFQT',
        'poly_svm_classifier.zip': 'https://drive.google.com/uc?export=download&id=1GXhRisL7W-Yt5IDaD28Y4Bwx9_DR2o9x',
        'poly_svm_classifier_lem.zip': 'https://drive.google.com/uc?export=download&id=1KQp0R1puiSnmfXrmh4316b6QZlzN7VvY'
        # 'fully_connected_nn.zip': ''
        # 'fully_connected_nn_lem.zip': ''
    }

    for entry in models_dict.items():
        url = entry[1]
        name = entry[0]
        file_path = base_model_folder + name
        if os.path.exists(file_path[0:-4] + '.pkl') is False:
            print("Downloading File:", entry[0])
            gdown.download(url=url, output=file_path)

            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path=base_model_folder)
            os.remove(file_path)
        else:
            print("File '" + str(name[0:-4] + '.pkl' + "' already exists"))
    
    print("Base Models Downloaded.\n\n")

def download_pretrained_bert_boosted_models():
    print("BERT-BOOSTED MODELS DOWNLOADING")
    base_model_folder = "./models/base/"
    models_dict = {
        # 'rf_classifier_100.zip': '',
        # 'rf_classifier_500.zip': '',
        'linear_svm_classifier.zip': 'https://drive.google.com/uc?export=download&id=1nTr8rBzaZ1G75qptrgA9heViJI1L6xKq',
        'poly_svm_classifier.zip': 'https://drive.google.com/uc?export=download&id=1mNlK6abbYiupbznVOdi3wXmYHt7X6ZVN'
        # 'fully_connected_nn.zip': ''
        # 'fully_connected_nn_lem.zip': ''
    }

    for entry in models_dict.items():
        url = entry[1]
        name = entry[0]
        file_path = base_model_folder + name
        if os.path.exists(file_path[0:-4] + '.pkl') is False:
            print("Downloading File:", entry[0])
            gdown.download(url=url, output=file_path)

            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path=base_model_folder)
            os.remove(file_path)
        else:
            print("File '" + str(name[0:-4] + '.pkl' + "' already exists"))
    
    print("Base Models Downloaded.\n\n")

### #### ###
### MAIN ###
### #### ###

download_datasets()
download_pretrained_base_models()
download_pretrained_bert_boosted_models()