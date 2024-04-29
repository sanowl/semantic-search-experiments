import math
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from dataclasses import dataclass

nltk.download('punkt')
nltk.download('stopwords')

# -------- Document Representation --------
@dataclass
class Document:
    text: str
    tokens: list[str] = None
    tfidf_vector: dict[str, float] = None

    def tokenize(self):
        """Preprocesses the text, removing stop words and stemming."""
        words = nltk.word_tokenize(self.text)
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        self.tokens = [stemmer.stem(w) for w in words if w.lower() not in stop_words]

    def calculate_tfidf(self, idf):
        """Calculates TF-IDF representation for this document with normalization."""
        tf = Counter(self.tokens)
        self.tfidf_vector = normalize_vector({term: tf[term] * idf.get(term, 0) for term in tf})

# -------- Vector Search Functions --------
def compute_idf(documents):
    """Computes the Inverse Document Frequency (IDF) over a set of documents."""
    N = len(documents)
    idf = {}
    for doc in documents:
        for term in doc.tokens:
            df = sum(term in doc.tokens for doc in documents)
            idf[term] = math.log(N / (df + 1)) 
    return idf

def normalize_vector(vector):
    """Normalizes a vector to have unit length."""
    magnitude = sum(value**2 for value in vector.values()) ** 0.5
    if magnitude == 0:  
        return vector  
    return {term: value / magnitude for term, value in vector.items()}

def cosine_similarity(vector1, vector2):
    """Calculates cosine similarity between two vectors."""
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1.values(), vector2.values()))
    magnitude1 = sum(v ** 2 for v in vector1.values()) ** 0.5
    magnitude2 = sum(v ** 2 for v in vector2.values()) ** 0.5
    return dot_product / (magnitude1 * magnitude2)  

# -------- Example Usage --------
if __name__ == "__main__": 
    documents = [
        Document("I love dogs. They are furry and playful."),
        Document("Cats are independent and curious creatures."),
        Document("My favorite outdoor activities are hiking and camping."),
        Document("Cooking is my passion. I enjoy experimenting.")
    ]

    # Preprocessing
    for doc in documents:
        doc.tokenize()

    # Compute IDF across documents
    idf = compute_idf(documents)

    # Calculate TF-IDF vectors
    for doc in documents:
        doc.calculate_tfidf(idf)

    # Query processing
    query = Document("I like animals that are playful.")
    query.tokenize()
    query.calculate_tfidf(idf)

    # Find similarities
    for i, doc in enumerate(documents):
        similarity = cosine_similarity(doc.tfidf_vector, query.tfidf_vector) 
        print(f"Similarity to Document {i}: {similarity:.3f}")
