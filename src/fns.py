# test function
def hello():
    return "Hello World!"



# preprocessing function
def text_preprocess(doc):
    
    preprocessed_doc = []
    stopwords = nltk.corpus.stopwords.words("english")
    
    for text in doc:
        
        # make lowercase
        text = text.lower()
        
        # remove urls
        text = tf.strings.regex_replace(text, "(?:https?)?:\/\/t.co\/\w*", " ")
        
        # remove mentions
        text = tf.strings.regex_replace(text, "@\w+", " ")
        
        # correct typos
        text = TextBlob(text.numpy().decode("utf-8")).correct().string
        
        # tokenize by word
        word_tokens = word_tokenize(text)
        
        # remove non-alphabetical characters
        word_tokens = [word for word in word_tokens if word.isalnum()]
        
        # remove stop words
        word_tokens = [word for word in word_tokens if word not in stopwords]
        
        # apply stemmer
        stemmer = PorterStemmer()
        word_tokens = [stemmer.stem(word) for word in word_tokens]
        
        # reappend to preprocessed doc
        preprocessed_doc.append(word_tokens)
    
    return preprocessed_doc




