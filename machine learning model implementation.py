# spam_detection_model.ipynb

# Importing required libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# import data from the user

data = {
    "label": ["spam", "ham", "ham", "spam", "ham"],
    "message": [
        "Congratulations! You've won a $1000 Walmart gift card. Click here to claim.",
        "Hey, are we still on for lunch today?",
        "Don't forget to bring the documents.",
        "URGENT! Your account has been compromised. Respond immediately.",
        "Can you send me the notes from today's meeting?"
    ]
}
df = pd.DataFrame(data)

# preprocessing the initilized data

df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to 0 and 1
X = df['message']
y = df['label']

# Converting the text into features

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Seperate the training set and testing set

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# initilize the predictions

y_pred = model.predict(X_test)

# Evaluate the model

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test,y_pred))
