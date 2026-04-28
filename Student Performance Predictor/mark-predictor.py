# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Load Dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Preview:")
    print(df.head())
    print("\nColumn Names:")
    print(df.columns)
    return df


# 2. Prepare Features and Target
def prepare_data(df):
    X = df[['Study Hours', 'Sleep Hours', 'Attendance (%)']]
    y = df['Grades']
    return X, y


# 3. Split Data
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\nTraining size:", X_train.shape)
    print("Testing size:", X_test.shape)
    return X_train, X_test, y_train, y_test


# 4. Train Model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\nModel trained successfully!")
    return model


# 5. Make Predictions
def make_predictions(model, X_test):
    y_pred = model.predict(X_test)
    print("\nPredicted values:")
    print(y_pred)
    return y_pred


# 6. Evaluate Model
def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print("Mean Absolute Error:", mae)
    print("R2 Score:", r2)


# 7. Custom Prediction
def custom_prediction(model, study, sleep, attendance):
    new_data = pd.DataFrame([[study, sleep, attendance]],
                            columns=['Study Hours', 'Sleep Hours', 'Attendance (%)'])

    prediction = model.predict(new_data)
    print(f"\nCustom Prediction ({study} hrs study, {sleep} hrs sleep, {attendance}% attendance):")
    print("Predicted Grade:", prediction[0])


# MAIN FUNCTION (Entry point)
def main():
    # Load
    df = load_data("marks-data.csv")

    # Prepare
    X, y = prepare_data(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train
    model = train_model(X_train, y_train)

    # Predict
    y_pred = make_predictions(model, X_test)

    # Evaluate
    evaluate_model(y_test, y_pred)

    # Custom Prediction
    custom_prediction(model, 5, 7, 80)


# Run the program
if __name__ == "__main__":
    main()