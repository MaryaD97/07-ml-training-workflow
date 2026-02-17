from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.preprocess import preprocess_data

def train_model():
    df = preprocess_data("wrong_path.csv")

    X = df.drop("bought", axis=1)
    y = df["bought"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {acc}")

if __name__ == "__main__":
    train_model()

