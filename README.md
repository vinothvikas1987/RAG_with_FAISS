This code is a vanilla RAG model and doesn't use any popular libraries like Langchain or LlamaIndex. A prompt with query will give a decent search for small and simple queries.It embeds the text,does a FAISS for cosine similar vectors and sends the embeds to a RAG model of Facebook,. It halluciantes its response based on the input.



#Chunking - Usage of LlamaIndex or Langchainfor sentence splitting or text splitting or Title extraction rather than conventional libraries like gensim,NTLK or spaCy is simpler and also equally powerful. 

#Vector DB - Indexing the embeds is one of the crucial task to get the most relevant answers and rather than brute force distance calculation, 
✅ FAISS
✅ Pinecone
✅ Weaviate
✅ NMSLIB
✅ Annoy (Approximate Nearest Neighbor)
✅ HNSW (Hierarchical Navigable Small World)
VecotrDB cant store text and embeds, so It can generate meaningful statements but with hallucination. It is useful if the intention is for faster search and not for reproducing facts from content.

#Vector Storage - 
✅ ChromaDB
✅ Milvus
✅ Qdrant
✅ Pinecone (with metadata)
✅ Weaviate (with hybrid search)
✅ AWS DocumentDB (with semantic search) can be used for storing the embeddings along with title and metadata values for cosine similarity findings.LlamaIndex/Llangchain supports most Vector of this. 

Llanchnain completed the whole pipeline, while Llamaindex needs one more step(model retrival) to close the pipelin

#Hierarchical retrieval - Different Documents summary are created and used for querying - Based on the relevant summary document is identified and top k chunks in the document is extracted

#Hypothetical Question - Let LLM create a hypothetical question for relevant index and use those questions to match with the query

#HyDE - Use LLM to generate response and use that response to find index match.

#Context Window - Increase the context window after retrieving the best sentence to few more sentences.

#Parent child for context window - This is similar to increasing context with sentence but at a more granular level, Out of all the relevant child chunks the mode value is used to select the parent chunk.
https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/
#Combine Classic with Neo - BM25 or IT-TDF is used for query along with vector index that was created already and use Reciprocal Rank Fusion to rerank the search index
PostProcess to get the best results. https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/
