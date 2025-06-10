import os
from pymilvus import MilvusClient
from .log import logger

def compact():
    logger.info("Starting compaction job")
    
    try:
        password = os.getenv("MILVUS_ROOT_PASSWORD")
        uri = os.getenv("MILVUS_URI")
        collection_name = os.getenv("MILVUS_COLLECTION_NAME")
        
        logger.info(f"Connecting to Milvus at {uri}")
        
        client = MilvusClient(
            uri=uri,
            token=f"root:{password}"
        )
        
        logger.info(f"Starting compaction for collection: {collection_name}")
        
        res = client.compact(
            collection_name=collection_name,
            is_clustering=False,
            timeout=None
        )
        logger.info(f"Compaction completed for collection: {res}")

        logger.info("Starting flush operation...")
        client.flush(
            collection_name=collection_name,
            timeout=None
        )
        logger.info(f"Flushed collection: {collection_name}")
        
        logger.info("Compaction job completed successfully")
        
    except Exception as e:
        logger.error(f"Error during compaction: {str(e)}", exc_info=True)
        raise
