from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from langchain_openai import OpenAIEmbeddings
from src.config import OPENAI_API_KEY, TEXT_EMBEDDING_MODEL, logger_factory
from src.metaculus import list_questions
from src.data_models.QuestionDetails import QuestionDetails
from langchain_core.documents import Document
from typing import Dict



class VectorStoreManager:
    """
    Class to update the vector store with the latest questions from Metaculus.

    The vector store is a database of questions and their embeddings, which can be used to search for similar questions.
    Only binary questions are added.

    Needs to run the following before importing this module:
        import sys
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    """

    def __init__(self):
        self.logger = logger_factory.make_logger("Vector Store Updater")

        embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY, model=TEXT_EMBEDDING_MODEL)
        persistent_client = chromadb.PersistentClient(
            path="data/chroma_langchain_db")
        collection = persistent_client.get_or_create_collection(
            "metaculus_questions",
            metadata={"hnsw:space": "cosine"})  # l2 is the default)

        self.vector_store = Chroma(
            client=persistent_client,
            embedding_function=embeddings,
        )

    def update_store(self):
        """
        Updates the vector store with the latest questions from Metaculus.
        
        Adds questions that are not already in the vector store, until no new questions are found.
        """
        existing_ids = [int(id_str) for id_str in self.vector_store.get(include=[]).get("ids")]
        current_offset = 0
        while True:
            ls = list_questions(tournament_id=None,
                                order_by="-publish_time",
                                offset=current_offset,
                                count=100,
                                status=None,
                                forecast_type="binary")
            if len(ls["results"]) == 0:
                self.logger.debug("No more questions found, stopping.")
                break
            question_details_list = [QuestionDetails(details_dict) for details_dict in ls["results"]]
            filtered_question_details_list = [qd for qd in question_details_list if qd.id not in existing_ids]
            new_document_list = [self._document_from_question_details(qd) for qd in filtered_question_details_list]
            if len(new_document_list) == 0:
                self.logger.debug("No new documents found, stopping.")
                break
            self.logger.info(f"Adding {len(new_document_list)} new documents to the vector store, going from {filtered_question_details_list[0].publish_date} to {filtered_question_details_list[-1].publish_date}")
            self.vector_store.add_documents(new_document_list)
            current_offset += 100
    
    def _metadata_from_question_details(self, qd: QuestionDetails) -> Dict:
        return {
        "question_id": qd.id,
        "created_date" : qd.created_date,
        "publish_date": qd.publish_date,
        "close_date": qd.close_date,
        "resolve_date" : qd.resolve_date,
        "created_timestamp" : qd.created_time.timestamp() if qd.created_time else None,
        "publish_timestamp" : qd.publish_time.timestamp() if qd.created_time else None,
        "close_timestamp" : qd.close_time.timestamp() if qd.created_time else None,
        "resolve_timestamp" : qd.resolve_time.timestamp() if qd.created_time else None,
        "resolve_timestamp" : qd.resolve_time.timestamp() if qd.created_time else None,
        "last_activity_date" : qd.last_activity_date,
        "forecast_type": qd.forecast_type,
        "project_ids": str(qd.project_ids),
    }

    def _document_from_question_details(self, qd: QuestionDetails) -> Document:
        return Document(
            page_content=qd.title,
            metadata=self._metadata_from_question_details(qd),
            id=qd.id,
        )