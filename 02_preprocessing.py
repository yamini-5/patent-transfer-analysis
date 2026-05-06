import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from tqdm import tqdm

# Download necessary NLTK datasets
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

def preprocess_dataframe(df, text_column='text'):
    """
    Applies text preprocessing to a specific column in the dataframe.
    """
    tqdm.pandas(desc="Preprocessing Text")
    df['clean_text'] = df[text_column].progress_apply(clean_text)
    return df

if __name__ == "__main__":
    sample_text = "This is a sample abstract about a Machine Learning algorithm network!"
    print("Original:", sample_text)
    print("Cleaned:", clean_text(sample_text))
