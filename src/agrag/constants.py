DOC_ID_KEY = "doc_id"
CHUNK_ID_KEY = "chunk_id"
DOC_TEXT_KEY = "text"
SOURCE_KEY = "source"
EMBEDDING_KEY = "embedding"
EMBEDDING_HIDDEN_DIM_KEY = "embedding_hidden_dim"
# Parent-child (hierarchical) chunking. ``parent_id`` links a small child chunk
# to a larger parent chunk in the parent store; ``parent_text`` is the parent
# chunk's text (kept in the separate parent-store DataFrame, not on child rows).
PARENT_ID_KEY = "parent_id"
PARENT_TEXT_KEY = "parent_text"
SUPPORTED_FILE_EXTENSIONS = [".pdf", ".txt", ".docx", ".doc", ".rtf", ".csv", ".md", ".py", ".log"]
SUPPORTED_HTML_TAGS = ["p", "table"]
EVALUATION_DIR = "./evaluation_data"
EVALUATION_MAX_FILE_SIZE = 5 * 1000 * 1000  # 5 MB
LOGGER_NAME = "AutoGluon-RAG-logger"
