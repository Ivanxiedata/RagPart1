import nest_asyncio
from loguru import logger
import asyncio
import time
from qdrant_client import AsyncQdrantClient

class Qdrant:
    def __init__(self, input_dir_path, input_question):
        self.input_dir_path = input_dir_path
        self.embed_model_name = "BAAI/bge-large-en-v1.5"
        self.rerank_model = "cross-encoder/ms-marco-MiniLM-L-2-v2"
        self.input_question = input_question  # List of queries
        self.topn = 10


        nest_asyncio.apply()

    def read_the_documents(self):
        """Reads PDF documents from the input directory."""
        from llama_index.core import SimpleDirectoryReader

        loader = SimpleDirectoryReader(
            input_dir=self.input_dir_path,
            required_exts=[".pdf"],
            recursive=True
        )
        docs = loader.load_data()
        logger.info(f'Docs info: {len(docs)}; {type(docs)}')
        return docs

    def set_up_Qdrant_DB(self):
        """Sets up a Qdrant database for storing vectors."""
        import qdrant_client
        collection_name = 'chat_with_docs'
        client = qdrant_client.QdrantClient(
            host='localhost',
            port=6333
        )
        logger.info('✅ Qdrant DB is set up locally')
        return collection_name, client



    def create_index_for_docs(self, documents):
        """Converts documents into vectors for fast retrieval."""
        from llama_index.core import VectorStoreIndex, StorageContext, Settings
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding


        collection_name, client = self.set_up_Qdrant_DB()


        try:
            if client is None:
                raise ValueError("❌ Error: Qdrant client is not initialized.")
            if not documents:
                raise ValueError("❌ Error: No docs provided.")

            embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name, trust_remote_code=True)
            Settings.embed_model = embed_model

            vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            logger.info('✅ Successfully generated indices for documents')
            return index

        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return None



    def setup_llm_prompt(self):
        """Creates a prompt template for LLM queries."""
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings, PromptTemplate

        llm = Ollama(model="llama3", request_timeout=120.0)
        Settings.llm = llm

        qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above, answer the query step by step. "
            "If you don't know, say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
        )

        return PromptTemplate(qa_prompt_tmpl_str)

    def re_rank_retrival_vectors(self):
        """Applies a reranker to reorder search results."""
        from llama_index.core.postprocessor import SentenceTransformerRerank

        return SentenceTransformerRerank(
            model=self.rerank_model,
            top_n=self.topn
        )

    def query_documents(self, index):
        """Synchronous query processing (one by one)."""
        qa_prompt_tmpl = self.setup_llm_prompt()
        rerank = self.re_rank_retrival_vectors()

        query_engine = index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[rerank],
        )

        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

        result_list = []
        for query in self.input_question:
            response = query_engine.query(query)
            result_list.append(response)

        return result_list

    async def query_documents_async(self, index, query):
        """Asynchronous query processing (parallel execution)."""
        qa_prompt_tmpl = self.setup_llm_prompt()
        rerank = self.re_rank_retrival_vectors()

        query_engine = index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[rerank],
        )

        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

        response = await query_engine.aquery(query)
        return response

    async def process_multiple_queries(self, index):
        """Executes multiple queries in parallel."""
        tasks = [self.query_documents_async(index, query) for query in self.input_question]
        return await asyncio.gather(*tasks)

def main():
    queries = [
        "What exactly is DSPy?",
        "How does Qdrant store vector embeddings?",
        "What is the purpose of reranking?",
        "Explain the difference between RAG and traditional search."
    ]

    qd = Qdrant(input_dir_path='docs', input_question=queries)
    documents = qd.read_the_documents()

    ########### 🚀 Non-Async Execution (Sequential) #############
    print("\n🔹 Running Non-Async Execution...")
    start_time = time.time()
    index = qd.create_index_for_docs(documents)
    result_sync = qd.query_documents(index)
    end_time = time.time()

    for query, response in zip(queries, result_sync):
        print(f"\n🔹 **Query:** {query}")
        print(f"🔸 **Response:** {response}\n")

    print(f"🕒 **Total Time (Sync): {end_time - start_time:.2f} seconds**")
    #


if __name__ == "__main__":
    main()
