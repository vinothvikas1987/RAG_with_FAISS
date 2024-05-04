
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
nltk.download('punkt')
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from transformers import pipeline
import torch

# !conda update -n base -c conda-forge conda -y
# !conda create --name faiss_1.8.0 -y
# !conda activate faiss_1.8.0
# !conda update -n base -c conda-forge conda -y

# !apt-get install -y poppler-utils

def pdf_to_text(pdf_path):
    pages = convert_from_path(pdf_path, 300)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text


pdf_path = ''

text = pdf_to_text(pdf_path)

len(text)

def chunk_text(text, chunk_size=100):

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks


chunk_size = 100
chunks = chunk_text(text, chunk_size)

#  !pip install transformers

from transformers import DPRContextEncoderTokenizerFast
context_token = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

context_tokens = context_token(chunks,truncation = True, padding = "longest" ,return_tensors="pt")

input_ids = context_tokens['input_ids']

input_ids[0]

from transformers import DPRContextEncoder
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

input_ids.size()[0]

import math
step = 0
batch = 10
num_passages = input_ids.size()[0]
num_batches = math.ceil(num_passages/batch)
embed_batches = []

for i in range(0,num_passages,batch):
    batch_ids = input_ids[i:i+10,:]
    output = context_encoder(batch_ids,return_dict = True)
    context_embeddings = output["pooler_output"]
    context_embeddings = context_embeddings.detach().numpy()
    embed_batches.append(context_embeddings)

import numpy as np
context_embeddings = np.concatenate(embed_batches,axis=0)

context_embeddings.shape

# !pip install faiss-cpu

import faiss
dim = 768
m=128
index = faiss.IndexHNSWFlat(dim,m,faiss.METRIC_INNER_PRODUCT)

index.train(context_embeddings)
index.add(context_embeddings)

context_embeddings.shape

#following code is used for debugging

# from transformers import DPRQuestionEncoder
# Q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")

# from transformers import DPRQuestionEncoderTokenizerFast
# Q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

# input_ids = Q_tokenizer.encode("what did mark twain say?",return_tensors = "pt")
# outputs = Q_encoder(input_ids)
# Q_embed = outputs["pooler_output"]

# Q_embed_numpy = Q_embed.detach().numpy()

# D,I = index.search(Q_embed_numpy,k=5)

# import textwrap
# wrap = textwrap.TextWrapper(width=10000)
# for i in I[0]:
#     print("index",i)
#     ans = chunks[i]
#     print("Answer:",wrap.fill(ans))

# len(chunks)

# chunk_corpus = {"title":[],"text" : chunks}

# chunk_corpus

from datasets import Dataset
import pandas as pd
df = pd.DataFrame(chunks)

df.head(3)

dataset = Dataset.from_pandas(df)

dataset_full = dataset.add_column("title", ["ignore this"] * len(dataset))

dataset_full = dataset_full.rename_column("0", "text")

context_embeddings.shape

embeddings_context = []
for i in range(context_embeddings.shape[0]):
    embeddings_context.append(context_embeddings[i,:])

dataset_full = dataset_full.add_column("embeddings",embeddings_context)

dataset_full

index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
dataset_full.add_faiss_index(column = "embeddings",index_name = 'embeddings', custom_index = index,faiss_verbose = True)

from transformers import RagRetriever
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    indexed_dataset = dataset_full,
    index_name="embeddings")

from transformers import RagTokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

from transformers import RagSequenceForGeneration
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever = retriever)

question = "what are the oranges used for?"
input_ids = tokenizer.question_encoder(question,return_tensors='pt')["input_ids"]

gen = model.generate(input_ids)
gen_string = tokenizer.batch_decode(gen,skip_special_tokens=True)[0]

print(gen_string)





