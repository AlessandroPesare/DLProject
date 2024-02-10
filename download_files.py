import gdown
import zipfile
import os

def download_datasets():
    print("DOWNLOADING DATASETS...")
    dataset_folder = "./datasets/"
    file_dict = {
        'jigsaw_train_set.csv': 'https://drive.google.com/uc?export=download&id=1DG3WQxA-Qx358k1bcGXtmkdwXMMGM4yF',
        'jigsaw_test_set.csv': 'https://drive.google.com/uc?export=download&id=14QHA6-99fIGqhQm-occTMvk00JNGhqnA',
        'training_set.csv': 'https://drive.google.com/uc?export=download&id=1ISAt0gktW6Wsy1_dAgXH55s4KGq7-efR',
        'training_set_lemmatized.csv': 'https://drive.google.com/uc?export=download&id=1livxNTJ1mfMd1_mqywBBtxgdY1_MyAMc',
        'test_set.csv': 'https://drive.google.com/uc?export=download&id=121vKZoda39x-cEY4Gsk3ojHBy52un3lQ',
        'test_set_lemmatized.csv': 'https://drive.google.com/uc?export=download&id=17dEAau_krrkQs7XsCZHc_f_9iaJ8wBYJ',
        'X_train_bert.csv': 'https://drive.google.com/uc?export=download&id=1gUfcu6MA5XelHrOzQeyzAm8iXTiktFei',
        'X_test_bert.csv': 'https://drive.google.com/uc?export=download&id=1HrbuyoDCMe_5f_fQEhji0e8a5mldRzD0'
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
    print("DOWNLOADING BASE MODELS...")
    model_folder = "./models/base/"
    models_dict = {
        'rf_classifier_500.zip': 'https://drive.google.com/uc?export=download&id=1cLryAR76cxh3085c4jJ6fYhetFCT6eI7',
        'rf_classifier_500_lem.zip': 'https://drive.google.com/uc?export=download&id=1lLwjArgj_IbNOIhVfXwO68j_8OYOWCHx',
        'rf_classifier_1000.zip': 'https://drive.google.com/uc?export=download&id=1QHQNg208o8azTExGy_IYDE0EX1tq8GSr',
        'rf_classifier_1000_lem.zip': 'https://drive.google.com/uc?export=download&id=1P3lp_v8Z1XjuMj4NqwwJ9j2KfwcqF_ft',
        'linear_svm_classifier.zip': 'https://drive.google.com/uc?export=download&id=1uQHITEILMyeOR4pCjlxd613DFMbpVYgq',
        'linear_svm_classifier_lem.zip': 'https://drive.google.com/uc?export=download&id=1hqnEpnow_5TEI0Bjcwn5vYORfSbrsi85'
    }

    for entry in models_dict.items():
        url = entry[1]
        name = entry[0]
        file_path = model_folder + name
        if os.path.exists(file_path[0:-4] + '.pkl') is False:
            print("Downloading File:", entry[0])
            gdown.download(url=url, output=file_path)

            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path=model_folder)
            os.remove(file_path)
        else:
            print("File '" + str(name[0:-4] + '.pkl' + "' already exists"))
    
    print("Base Models Downloaded.\n\n")

def download_pretrained_bert_boosted_models():
    print("DOWNLOADING BERT-BOOSTED MODELS...")
    model_folder = "./models/bert_boosted/"
    models_dict = {
        'rf_classifier_500.zip': 'https://drive.google.com/uc?export=download&id=1mKYh7weikxT1r_QPvAkGB7HSuXp9yCfo',
        'rf_classifier_1000.zip': 'https://drive.google.com/uc?export=download&id=1WCtigFIN6W3y7gWOfAAEAGzNwP1MOks4',
        'linear_svm_classifier.zip': 'https://drive.google.com/uc?export=download&id=1OASD0wt5HaaF9vob5BE6T5KsVSmmK_m0',
    }

    for entry in models_dict.items():
        url = entry[1]
        name = entry[0]
        file_path = model_folder + name
        if os.path.exists(file_path[0:-4] + '.pkl') is False:
            print("Downloading File:", entry[0])
            gdown.download(url=url, output=file_path)

            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(path=model_folder)
            os.remove(file_path)
        else:
            print("File '" + str(name[0:-4] + '.pkl' + "' already exists"))
    
    print("BERT-Boosted Models Downloaded.\n\n")

def download_pretrained_base_nn():
    print("DOWNLOADING BASE NEURAL NETWORKS...")
    model_folder = "./models/base/"
    models_dict = {
        'neural_classifier_big.h5': 'https://drive.google.com/uc?export=download&id=1DHvrQComtdexLZcSZ0MfX4XGHVTovEcP',
        'neural_classifier_big_lem.h5': 'https://drive.google.com/uc?export=download&id=112KKBxnFrtm0uUXaxbc81WWFt-56zcW0',
        'neural_classifier_small.h5': 'https://drive.google.com/uc?export=download&id=1OayRWx7I20NlEDNpjNobpiGB_6MSkXUv',
        'neural_classifier_small_lem.h5': 'https://drive.google.com/uc?export=download&id=1cYIaTLz7I1xDzXoj8IZEZip_bBxmfnik'
    }

    for entry in models_dict.items():
        url = entry[1]
        name = entry[0]
        file_path = model_folder + name
        if os.path.exists(file_path) is False:
            print("Downloading File:", entry[0])
            gdown.download(url=url, output=file_path)
        else:
            print("File '" + str(name + "' already exists"))
    
    print("Base Neural Network Classifiers Downloaded.\n\n")

def download_pretrained_bert_boosted_nn():
    print("DOWNLOADING BERT-BOOSTED NEURAL NETWORKS...")
    model_folder = "./models/bert_boosted/"
    models_dict = {
        'neural_classifier_big.h5': 'https://drive.google.com/uc?export=download&id=1LR1rrgffYhkPxGPMPUgtU8PdznehWBR5',
        'neural_classifier_small.h5': 'https://drive.google.com/uc?export=download&id=14yDwvINf3I_XS0aj1sL9ppdBJOxwSZl-'
    }

    for entry in models_dict.items():
        url = entry[1]
        name = entry[0]
        file_path = model_folder + name
        if os.path.exists(file_path) is False:
            print("Downloading File:", entry[0])
            gdown.download(url=url, output=file_path)
        else:
            print("File '" + str(name + "' already exists"))
        
    print("BERT-Boosted Neural Network Classifiers Downloaded.\n\n")

### #### ###
### MAIN ###
### #### ###

download_datasets()
download_pretrained_base_models()
download_pretrained_bert_boosted_models()
download_pretrained_base_nn()
download_pretrained_bert_boosted_nn()