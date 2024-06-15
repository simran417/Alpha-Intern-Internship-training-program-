import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download NLTK stopwords and punkt 
# nltk.download('stopwords')
# nltk.download('punkt')


df = pd.read_csv('C:/Users/hp/OneDrive/Documents/spam_dataset.csv')

# Display the first few rows and column names
# print(df.head())
# print(df.columns)

# Keep only necessary columns and drop missing values
df = df[['label', 'text']]
df.dropna(inplace=True)

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text


df['text'] = df['text'].apply(preprocess_text)

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def remove_stopwords_and_stem(text):
    words = text.split()
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

df['text'] = df['text'].apply(remove_stopwords_and_stem)

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(subset=['label'], inplace=True)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display evaluation metrics
(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'F1 Score: {f1}')


def classify_text(user_text):
    processed_text = preprocess_text(user_text)
    processed_text = remove_stopwords_and_stem(processed_text)

    vectorized_text = vectorizer.transform([processed_text])
    
    prediction = model.predict(vectorized_text)
    
    return 'spam' if prediction[0] == 1 else 'important'


user_input = input("Enter a message to classify: ")
result = classify_text(user_input)
print(f'The message is classified as: {result}')
