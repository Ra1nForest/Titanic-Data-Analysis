import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    df = pd.read_csv('train.csv')
    return df

def clean_data(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    target = 'Survived'
    
    return df[features], df[target]

def run_pipeline():
    print("--- Starting Titanic Data Pipeline ---")
    
    raw_df = load_data()
    print(f"Loaded {len(raw_df)} rows of data.")

    X, y = clean_data(raw_df)
    print("Data cleaning completed.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model Training Finished!")
    print(f"✅ Validation Accuracy: {accuracy:.2f} ({(accuracy*100):.1f}%)")
    print("--- Pipeline Finished Successfully ---")

if __name__ == "__main__":
    run_pipeline()