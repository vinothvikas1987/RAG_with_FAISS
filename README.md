This code is a vanilla RAG model and doesn't use any popular libraries like Langchain or LlamaIndex. A prompt with query will give a decent search for small and simple queries.
#Chunking - Usage of LlamaIndex or Langchainfor sentence splitting or text splitting or Title extraction rather than conventional libraries like gensim,NTLK or spaCy is simpler and also equally powerful. 
#Vector DB - Indexing the vectors is one of the crucial task to get the most relevant answers and rather than brute force distance calculation use of AWS Document DB,FAISS,NMSLIB,ANNOY -HNSW ANN gets us optimal results.
#Vector Storage - Pinecone,Weaveite,Chroma can be used for storing the embeddings along with title and metadata values for cosine similarity findings.LlamaIndex supports most Vector DB.
#Hierarchical retrieval - Different Documents summary are created and used for querying - Based on the relevant summary document is identified and top k chunks in the document is extracted
#Hypothetical Question - Let LLM create a hypothetical question for relevant index and use those questions to match with the query
#HyDE - Use LLM to generate response and use that response to find index match.
#Context Window - Increase the context window after retrieving the best sentence to few more sentences.
#Parent child for context window - This is similar to increasing context with sentence but at a more granular level, Out of all the relevant child chunks the mode value is used to select the parent chunk.
https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/
#Combine Classic with Neo - BM25 or IT-TDF is used for query along with vector index that was created already and use Reciprocal Rank Fusion to rerank the search index
PostProcess to get the best results. https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/
