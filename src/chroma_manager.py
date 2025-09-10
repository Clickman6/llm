from config import CHROMA_HOST, CHROMA_PORT

from datasets import load_dataset

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TokenTextSplitter

class ChromaManager:
    def __init__(
        self,
        embeddings: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        collection_name: str = "terraria",
        batch_size: int = 2500,
        host: str = CHROMA_HOST,
        port: int = CHROMA_PORT,

        dataset: str = 'lparkourer10/terraria-wiki'
    ):
        self.splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=32)

        self.embeddings = HuggingFaceEmbeddings(model=embeddings)
        self.collection_name = collection_name
        self.batch_size = batch_size
    
        self.settings = Settings(
            chroma_api_impl="chromadb.api.fastapi.FastAPI",
            chroma_server_host=host,
            chroma_server_http_port=str(port),
        )

        self.vectordb = Chroma(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            client_settings=self.settings,
        )

        if dataset and self.is_empty():
            self.insert(self.load(dataset))

    def is_empty(self) -> bool:
        data = self.vectordb.get(include=["metadatas"])["metadatas"]
        return len(data) == 0
    
    def insert(self, dataset) -> None:
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            documents = self.splitter.create_documents(texts=batch['question'], metadatas=[{"answer": a} for a in batch['answer']])
            self.vectordb.add_documents(documents)

    def load(self, name: str, split: str = 'train'):
        return load_dataset(name, split=split)

    def retriever(self, count: int = 5):
        return self.vectordb.as_retriever(search_kwargs={"k": count})
