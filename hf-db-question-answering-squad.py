import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, pipeline
from pgvector.psycopg2 import register_vector
import psycopg2
from datasets import load_dataset


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

    # Create the vector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    register_vector(conn)

    # Create a table to store embeddings and content
    # The embedding column is defined as a vector(384) column.
    #
    # The number 384 is used here to specify the dimensionality of the vector in the embedding vector(384) column.
    # This means that each embedding stored in this column is a vector with 384 dimensions.
    # The dimensionality is determined by the embedding model being used,
    # which in this case is "sentence-transformers/all-MiniLM-L6-v2". This model produces embeddings with
    # 384 dimensions.
    cursor.execute("CREATE TABLE IF NOT EXISTS qa_embeddings (id SERIAL PRIMARY KEY, content TEXT, embedding vector(384))")
    cursor.execute("CREATE TABLE IF NOT EXISTS qa_squad_status (id SERIAL PRIMARY KEY, loaded BOOLEAN)")

    return cursor


def populate_database(cursor, dataset):
    """
    Populate the database with embeddings from the dataset
    :param cursor: The database cursor
    :param dataset: The dataset to populate the database with
    """

    cursor.execute("SELECT loaded FROM qa_squad_status")
    loaded = cursor.fetchone()

    if not loaded:
        print(f"Loading {max_examples} examples from squad dataset into the database")

        idx = 0
        for data in dataset:
            if idx == max_examples:
                cursor.execute("INSERT INTO qa_squad_status (loaded) VALUES (TRUE)")
                break

            text = data['context']
            if content_is_existing(cursor, text):
                continue

            embedding = get_embedding(text)
            cursor.execute("INSERT INTO qa_embeddings (content, embedding) VALUES (%s, %s)", (text, embedding))

            idx += 1

        print("Finished loading examples into the database")


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
    cursor.execute("SELECT content, embedding <=> %s AS distance FROM qa_embeddings ORDER BY distance LIMIT 5",
                   (question_embedding,))
    relevant_contexts = cursor.fetchall()

    results = []

    # Extract answers from relevant contexts
    for context in relevant_contexts:
        result = qa_pipeline(question=question, context=context[0])
        result["question"] = question
        result["context"] = context[0]
        results.append(result)

    return results


# Load the SQuAD 2.0 dataset
squad_dataset = load_dataset("squad_v2")
# Maximum number of examples to load
max_examples = 1000

# Load Hugging Face models
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

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

# Enable autocommit
conn.autocommit = True

try:
    # Example usage
    questions = [
        "Who wrote 'To kill a mocking bird'?",
        "Who is the founder of Apple company?"
    ]

    cur = prepare_database(conn)

    # Populate the database with the SQuAD dataset
    # This step will try to load the whole SQuAD dataset into the database.
    populate_database(cur, squad_dataset['train'])

    try:
        final_results = {}
        for ques in questions:
            results = answer_question(cur, ques)
            for result in results:
                if result['score'] < 0.5:  # Threshold for unanswerable questions
                    print(f"Question: {result['question']}")
                    print("[Answer: This question is unanswerable based on the given context.]")
                    print(f"Context: {result['context']}")
                    print(f"Score: {result['score']:.4f}")
                else:
                    if result['question'] not in final_results:
                        final_results[result['question']] = result['answer']

                    print(f"Question: {result['question']}")
                    print(f"Answer: {result['answer']}")
                    print(f"Context: {result['context']}")
                    print(f"Score: {result['score']:.4f}")

                print("-" * 50)

        print("\n\n----[Final Answers]-------------------------------")
        for key, value in final_results.items():
            print(f"Question: {key}")
            print(f"Answer: {value}")
            print("-" * 50)
    finally:
        cur.close()

finally:
    # Close the database connection
    conn.close()
