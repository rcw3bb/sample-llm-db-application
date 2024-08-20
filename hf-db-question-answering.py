import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, pipeline
from pgvector.psycopg2 import register_vector
import psycopg2


def get_embedding(text):
    """
    Get the embedding of the text
    :param text: The text to embed
    :return: The embedding of the text
    """

    # The max_length parameter is used to specify the maximum number of tokens that the model supports.
    max_length = 256
    inputs = embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def connect_to_postgres(dbname, user, password, host, port):
    """Connects to a PostgreSQL database and returns a connection object."""

    try:
        connection = psycopg2.connect(
            database=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        print("Connected to PostgreSQL database successfully")
        return connection
    except psycopg2.Error as e:
        print("Error connecting to PostgreSQL database:", e)
        return None


def prepare_database(connection):
    """
    Prepare the database by creating a table to store embeddings and inserting sample data
    :param connection: The connection object.
    :return: The cursor of the database.
    """

    cursor = connection.cursor()

    # Create a table to store embeddings and content
    # The embedding column is defined as a vector(384) column.
    #
    # The number 384 is used here to specify the dimensionality of the vector in the embedding vector(384) column.
    # This means that each embedding stored in this column is a vector with 384 dimensions.
    # The dimensionality is determined by the embedding model being used,
    # which in this case is "sentence-transformers/all-MiniLM-L6-v2". This model produces embeddings with
    # 384 dimensions.
    cursor.execute("CREATE TABLE IF NOT EXISTS qa_embeddings (id SERIAL PRIMARY KEY, content TEXT, embedding vector(384))")

    # Sample data (longer passages for QA)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. The fox is known for its agility and speed, while the dog is often characterized by its relaxed nature.",
        "A journey of a thousand miles begins with a single step. This proverb emphasizes the importance of starting and perseverance in achieving long-term goals.",
        "To be or not to be, that is the question. This famous line from Shakespeare's Hamlet reflects on the philosophical nature of existence and the challenges of life.",
    ]

    # Insert sample data and embeddings
    for text in sample_texts:
        if content_is_existing(cursor, text):
            print("Content already exists in the database:", text)
            continue

        embedding = get_embedding(text)
        cursor.execute("INSERT INTO qa_embeddings (content, embedding) VALUES (%s, %s)", (text, embedding))

    connection.commit()
    return cursor


def content_is_existing(cursor, text):
    """
    Check if the content already exists in the database
    :param cursor: The database cursor
    :param text: The text to check
    :return: True if the content exists, False otherwise
    """

    cursor.execute("SELECT EXISTS(SELECT 1 FROM qa_embeddings WHERE content = %s)", (text,))
    return cursor.fetchone()[0]


def answer_question(cursor, question):
    """
    Answer a question using the QA model and similarity search
    :param cursor: The database cursor
    :param question: The question to answer
    :return: The response to the question
    """

    question_embedding = get_embedding(question)

    # Perform similarity search
    cursor.execute("SELECT content, embedding <=> %s AS distance FROM qa_embeddings ORDER BY distance LIMIT 1",
                   (question_embedding,))
    result_set = cursor.fetchone()
    most_similar_text = result_set[0]
    distance = result_set[1]

    # Use question-answering model
    qa_result = qa_pipeline(question=question, context=most_similar_text)

    return {
        "question": question,
        "answer": qa_result["answer"],
        "score": qa_result["score"],
        "context": most_similar_text,
        "similarity_distance": distance
    }


# Load Hugging Face models
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
qa_model_name = "distilbert-base-cased-distilled-squad"

embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# Create question-answering pipeline
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

# Connect to PostgreSQL
conn = connect_to_postgres(
    dbname="llmvector",
    user="postgres",
    password="mypassword",
    host="localhost",
    port="5432"
)

register_vector(conn)
try:
    # Example usage
    questions = [
        "What animal jumps over the dog?",
        "What does the proverb say about a journey?",
        "Who wrote 'To be or not to be'?"
    ]

    cur = prepare_database(conn)

    try:
        for ques in questions:
            result = answer_question(cur, ques)
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"Confidence Score: {result['score']:.4f}")
            print(f"Context: {result['context']}")
            print(f"Similarity Distance: {result['similarity_distance']:.4f}")
            print("---")
    finally:
        cur.close()

finally:
    # Close the database connection
    conn.close()
