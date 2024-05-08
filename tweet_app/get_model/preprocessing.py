from get_model.parameters import Params

import re
import contractions
import nltk
import torch
from transformers import DebertaTokenizer


# preprocessing function
def preprocess_text(
        doc: list[str],
        params: Params
) -> torch.Tensor:
    
    # list of ids and attention masks
    encoded_list = []
    
    # iterate over tweets
    for text in doc:
        
        # make lowercase
        text = text.lower()

        # remove urls
        text = re.sub(r"(?:https?)?:\/\/t.co\/\w*", " ", text)
        
        # remove twitter handles
        text = re.sub(r"@\w+", " ", text)

        # remove emoji pattern
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        
        text = emoji_pattern.sub(r'', text)

        # remove contractions
        word_list = [contractions.fix(word) for word in text.split()]
        
        # remove non-alphabetical characters
        word_list = [word for word in word_list if word.isalnum()]
        
        # remove stop words
        stopwords = nltk.corpus.stopwords.words("english")
        word_list = [word for word in word_list if word not in stopwords]

        # join to single string
        cleaned_tweet = " ".join(word_list)

        # get deberta tokenizer
        tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

        # apply Deberta tokenizer
        encoded_text = tokenizer(
            text = cleaned_tweet,
            max_length = params.max_len,
            padding = "max_length",
            truncation = True,
            return_attention_mask=True,
        )
        
        # make list of input ids and attention masks paired together
        encoded_list.append(encoded_text["input_ids"] + encoded_text["attention_mask"])
    
    return torch.tensor(encoded_list)




