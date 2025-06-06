import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

def load_data():
    df = pd.read_csv("adult_sample.csv")
    return df

def preprocess_data(df):
    df = df.copy()
    le = LabelEncoder()

    # Encode categorical columns
    for column in df.select_dtypes(include=['object']).columns:
        if column != 'income':
            df[column] = le.fit_transform(df[column].astype(str))
    
    # Encode target column
    df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
    return df

def train_model():
    df = load_data()
    df = preprocess_data(df)
    
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    model_path = "income_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    if os.path.exists(model_path):
        print(f"✅ Model saved successfully as {model_path}")
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {accuracy:.2f}")
    return model, accuracy

# Run training
train_model()

