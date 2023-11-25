import re

# Define a list of stopwords
STOPWORDS = {
    'the', 'and', 'is', 'in', 'to', 'of', 'it', 'you', 'that', 'a', 'i', 'for', 'on', 'with', 'this',
    'as', 'have', 'but', 'not', 'be'
}

# Other unused stopwords - can be added into stopwords as seen fit
UNUSED_STOPWORDS = {
    'were', 'then', "needn't", "wasn't", 'isn', 'just', 'we', 'yourselves', 'more', 'herself', 'wouldn',
    'aren', "mightn't", 'did', 'don', 'ma', "haven't", 'its', 'only', 'too', 'd', "hasn't", 'was', 'myself',
    'shan', 'other', 'our', 'again', 'each', 'yours', 'me', 'some', 'themselves', 'why', 'than', 'do', 'weren',
    'been', 'few', 'having', "she's", 'who', "you're", 'over', "isn't", 'nor', 'am', 'doesn', 'below', "shan't",
    'does', 'so', 'y', "that'll", 'haven', 'mustn', 'these', 'm', 'him', 'are', 'those', 'out', 'most', "you'll",
    'under', 't', 'has', 'up', 'should', 'both', 'no', 'he', 'hadn', 're', 'yourself', 'an', 'during', 'until',
    'between', "don't", 'into', "didn't", 'here', 'shouldn', 'ain', 'll', 'hers', "weren't", 'wasn', 'couldn',
    'which', "couldn't", 'their', 'where', 'how', 'whom', 'same', 'or', 'can', 'didn', 'while', 'at', "hadn't",
    'own', 'needn', 'before', 'such', 'because', 'from', 'if', 'itself', 'after', 'ourselves', "it's", "should've",
    'mightn', "mustn't", 'theirs', 'when', 'all', 'about', 'will', 'being', 'above', 'ours', 'them', 'her', 'there',
    'very', 'hasn', 'down', 'further', "won't", 'his', 'what', 'doing', 'any', "you've", 'now', 'they', 'won', 
    'your', 'through', "aren't", 'she', 'my'
}


# Define regexps
CONTRACTIONS_RE = re.compile(r"'|-|\.|!")
SYMBOLS_RE = re.compile(r"[^A-Za-z0-9\s]")
SPACES_RE = re.compile(r"\s+")

# Define dictionary for contraction replacements
CONTRACTIONS_DICT = {
    "what's": "what is",
    "n't": " not",
    "i'm": "i am", 
    "'re": " are",
    "'ve": " have", 
    "'d": " would",
    "'ll": " will"
}

# Cleanup data
def preprocess(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove contractions
    text = CONTRACTIONS_RE.sub(lambda match: CONTRACTIONS_DICT.get(match.group(0), match.group(0)), text)

    # Remove punctuation and symbols
    text = SYMBOLS_RE.sub(" ", text)

    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in STOPWORDS])

    # Remove extra spaces
    text = SPACES_RE.sub(" ", text).strip()

    return text