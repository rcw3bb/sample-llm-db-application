# Sample LLM DB Application

## Overview

This project demonstrates a sample application that utilizes a Large Language Model (LLM) for question answering, combined with a vector database for efficient similarity search. The application uses Hugging Face models for embeddings and question answering, and PostgreSQL with the `pgvector` extension for storing and querying embeddings.

## Features

- **Text Embedding**: Converts text into vector embeddings using a pre-trained model.
- **Similarity Search**: Finds the most similar text in the database to a given query using vector similarity.
- **Question Answering**: Answers questions based on the most similar text found in the database.

## Requirements

- Python 3.12.x
- Docker

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/rcw3bb/sample-llm-db-application.git
    cd sample-llm-db-application
    ```

2. **Install dependencies**:
    ```sh
    poetry install
    ```

3. **Set up PostgreSQL**:
    - Run the following command to start the PostgreSQL service:
        ```sh
        docker compose up -d
        ```
    - Update the connection details in the script if necessary.

## Usage

1. **Load Hugging Face models**:
    The script loads the following models:

    - Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
    - QA model: `bert-large-uncased-whole-word-masking-finetuned-squad`

2. **Insert sample data and embeddings**:
    The script inserts predefined sample texts and their embeddings into the database.

3. **Answer questions**:
    The script answers predefined questions by finding the most similar text in the database and using the QA model to generate answers.

4. **Run the script**:

    ```sh
    python hf-db-question-answering.py
    ```

    > Run the following for a subset of the SQuAD v2 dataset as sample data.
    >
    > ```sh
    > python hf-db-question-answering-squad.py
    > ```

## Example

The script will output answers to the predefined questions along with the confidence score and the context from which the answer was derived. The final answers are based on the first score that is equal to or greater than 0.5.

## Author

Ronaldo Webb