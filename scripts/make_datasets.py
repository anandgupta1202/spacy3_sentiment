# %%
%%time
# # Setup import files
from spacy.tokens import DocBin
import sys
import logging
from logging.config import fileConfig
import os
import spacy
from datasets import load_dataset
import pandas as pd

# %%
# # Setup Logger
fileConfig("../logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)
logger.info("Reading Tweet Testset")

# %%
%%time
# # Load the spacy pretrained model
nlp = spacy.load("en_core_web_lg")

# %%
def make_docs(data):
    # # Make the dataset into tuple format - needed by Spacy
    data = data[:]
    data = list(data.itertuples(index=False, name=None))

    # # Stream the data into "nlp" object to return doc, context
    # # Add "cats" to doc object
    Docs = []
    for doc, context in nlp.pipe(data, as_tuples=True):
        # print(doc)
        if context == 0:
            doc.cats["negative"] = 1
            doc.cats["neutral"] = 0
            doc.cats["positive"] = 0
        elif context == 1:
            doc.cats["negative"] = 0
            doc.cats["neutral"] = 1
            doc.cats["positive"] = 0
        elif context == 2:
            doc.cats["negative"] = 0
            doc.cats["neutral"] = 0
            doc.cats["positive"] = 1
        Docs.append(doc)
    return Docs


def save_to_disk(data, file_path):
    docs = make_docs(data)
    # # Assign it to DocBin and save to disk
    # # Saving the binary document as train.spacy
    db = DocBin(docs=docs)
    db.to_disk(file_path)


# %%
# # Load the dataset from Huggingface
train_data = load_dataset("tweet_eval", "sentiment", split="train")
train_data.set_format("pandas")
save_to_disk(train_data, "../data/sentiment_train.spacy")
logger.info("Saved train set to disk")
# %%
val_data = load_dataset("tweet_eval", "sentiment", split="validation")
val_data.set_format("pandas")
save_to_disk(val_data, "../data/sentiment_val.spacy")
logger.info("Saved validation set to disk")

# %%
test_data = load_dataset("tweet_eval", "sentiment", split="test")
test_data.set_format("pandas")
save_to_disk(test_data, "../data/sentiment_test.spacy")
logger.info("Saved test set to disk")


