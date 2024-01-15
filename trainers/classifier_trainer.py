from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class ClassifierTrainer:
    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
