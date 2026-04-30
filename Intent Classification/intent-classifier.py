import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data(path):
    return pd.read_csv(path)


def preprocess(text_series):
    return text_series.str.lower().apply(lambda x: re.sub(r'[^a-zA-Z ]', '', x))


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def vectorize(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return vectorizer, X_train_vec, X_test_vec


def train_model(X_train_vec, y_train):
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    return model


def evaluate(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    return accuracy_score(y_test, y_pred)


def predict_intent(model, vectorizer, text):
    text = re.sub(r'[^a-zA-Z ]', '', text.lower())
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


def main():
    data = load_data("intent_dataset.csv")

    data['text'] = preprocess(data['text'])

    X = data['text']
    y = data['intent']

    X_train, X_test, y_train, y_test = split_data(X, y)

    vectorizer, X_train_vec, X_test_vec = vectorize(X_train, X_test)

    model = train_model(X_train_vec, y_train)

    acc = evaluate(model, X_test_vec, y_test)
    print("Accuracy:", acc)

    while True:
        user_input = input("Enter text (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        print("Intent:", predict_intent(model, vectorizer, user_input))


if __name__ == "__main__":
    main()