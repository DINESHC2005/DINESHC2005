import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample: Exploring the truth with advanced fake news detection powered by NLP

# Load dataset
df = pd.read_csv('news.csv')  # Assumes a CSV file with 'text' and 'label' columns
df = df[['text', 'label']]    # 'label' should be 'FAKE' or 'REAL'

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform training data, transform test data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize and train classifier
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfidf_train, y_train)

# Predict and evaluate
y_pred = classifier.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print(f"Accuracy: {round(score*100, 2)}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
