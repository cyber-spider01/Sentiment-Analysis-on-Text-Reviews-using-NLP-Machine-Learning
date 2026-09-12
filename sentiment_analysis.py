import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


data = {
    "review": [
        "Absolutely love this product! The quality is outstanding and durable.",
        "Worst customer service ever. Arrived completely broken and late.",
        "Works really well, easy to set up and very intuitive to use.",
        "Waste of money. The battery died after only two days of light use.",
        "Exceeded my expectations! Fast shipping and works flawlessly.",
        "Terrible experience. Does not match the description at all.",
        "Great value for the price. Highly recommend to everyone!",
        "Defective out of the box, total disappointment. Returning it immediately."
    ],
    "sentiment": [
        "Positive", "Negative", "Positive", "Negative",
        "Positive", "Negative", "Positive", "Negative"
    ]
}

df = pd.DataFrame(data)


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

stop_words.discard("not")
stop_words.discard("no")

def clean_text(text: str) -> str:
   
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    # Convert to lowercase and tokenize by whitespace
    tokens = text.lower().split()
 
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    return " ".join(cleaned_tokens)


df["cleaned_review"] = df["review"].apply(clean_text)


X = df["cleaned_review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print(classification_report(y_test, y_pred))


new_reviews = [
    "The build is solid and customer support helped me immediately!",
    "Do not buy this, it stopped working within a week."
]

cleaned_new = [clean_text(r) for r in new_reviews]
new_vec = vectorizer.transform(cleaned_new)
predictions = model.predict(new_vec)
probabilities = model.predict_proba(new_vec)

for rev, pred, prob in zip(new_reviews, predictions, probabilities):
    pos_idx = list(model.classes_).index("Positive")
    print(f"Review: \"{rev}\"")
    print(f"--> Sentiment: {pred} ({prob[pos_idx]:.2%} Positive Confidence)\n")