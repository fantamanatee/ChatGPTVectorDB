"""
Usage:
    python script_name.py /path/to/conversations.json 
"""

import unicodedata
import json
import re
import argparse
import os
from datetime import datetime
from pathlib import Path
from util import connect_to_db
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter



def extract_message_parts(message):
    """
    Extract the text parts from a message content.

    Args:
        message (dict): A message object.

    Returns:
        list: List of text parts.
    """
    content = message.get("content")
    if content and content.get("content_type") == "text":
        return content.get("parts", [])
    return []

def get_author_name(message):
    """
    Get the author name from a message.

    Args:
        message (dict): A message object.

    Returns:
        str: The author's role or a custom label.
    """
    author = message.get("author", {}).get("role", "")
    if author == "assistant":
        return "ChatGPT"
    elif author == "system":
        return "Custom user info"
    return author

def get_conversation_messages(conversation):
    """
    Extract messages from a conversation.

    Args:
        conversation (dict): A conversation object.

    Returns:
        list: List of messages with author and text.
    """
    messages = []
    current_node = conversation.get("current_node")
    mapping = conversation.get("mapping", {})
    while current_node:
        node = mapping.get(current_node, {})
        message = node.get("message") if node else None
        if message:
            parts = extract_message_parts(message)
            author = get_author_name(message)
            if parts and len(parts) > 0 and len(parts[0]) > 0:
                if author != "system" or message.get("metadata", {}).get(
                    "is_user_system_message"
                ):
                    messages.append({"author": author, "text": parts[0]})
        current_node = node.get("parent") if node else None
    return messages[::-1]

# def create_conversations_table(conn, table_name):
#     """
#     create a table using the connection. The columns are: 
#         title (text)
#         update_time (timestamp)
#         messages (text)
#     """
#     with conn.cursor() as cur:
#         cur.execute("""
#             SELECT EXISTS (
#                 SELECT 1
#                 FROM information_schema.tables 
#                 WHERE table_name = table_name
#             );
#         """)
#         if not cur.fetchone()[0]:
#             cur.execute("""
#                 CREATE TABLE conversations_chunk384 (
#                     id UUID PRIMARY KEY,
#                     title TEXT,
#                     update_time TIMESTAMP,
#                     messages TEXT,
#                     conversation_id UUID
#                 );
#             """)
#             conn.commit()

def insert_document_to_db(conn, table_name, document, updated_date, title, conversation_id):
    """
    Write document to a postgres table

    Args:
        table_name (str): The name of the table to insert messages into.
        document (str): a chunk of a conversation
        updated_date (datetime): The most recent update date of the conversation.
        title (str): The title of the conversation.
        conversation_id (str): The id of the conversation.
    """


    # Assuming you have a database connection `conn`
    with conn.cursor() as cursor:
        insert_query = f"INSERT INTO {table_name} (title, update_time, messages, conversation_id) VALUES (%s, %s, %s, %s)"
        cursor.execute(insert_query, (title, updated_date, document, conversation_id))
    conn.commit()

def write_conversations(conversations_data, table_name, chunk_size = -1):
    """
    Write conversation messages to text files and create a conversation summary JSON file.

    Args:
        conversations_data (list): List of conversation objects.
        output_dir (Path): Directory to save the output files.

    Returns:
        list: Information about created directories and files.
    """

    for conversation in tqdm(conversations_data):
        updated = conversation.get("update_time")
        if not updated:
            continue

        updated_date = datetime.fromtimestamp(updated)
        title = conversation.get("title", "Untitled")
        conversation_id = conversation.get("id")

        messages = get_conversation_messages(conversation) 
        concatenated_messages = "\n".join(
            f"{message['author']}\n{message['text']}" for message in messages
        )

        if chunk_size != -1:
            # 6 char per word, 384 word max --> 6 * 384 = 2304
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=0)
            chunks = text_splitter.create_documents([concatenated_messages])
            chunks = [c.page_content for c in chunks]
        else: 
            chunks = [concatenated_messages]


        for chunk in chunks:
            insert_document_to_db(conn, table_name, chunk, updated_date, title, conversation_id)

def main():
    """
    Main function to parse arguments and process the conversations.
    """

    parser = argparse.ArgumentParser(
        description="Process conversation data from a JSON file."
    )
    parser.add_argument(
        "--input_file", 
        type=Path, 
        help="Path to the input conversations JSON file."
    )
    parser.add_argument(
        "--table_name",
        type=str,
        help="Name of the table to store the conversation data.",
    )
    parser.add_argument('--chunk_size', 
                        type=int,
                        default=-1,
                        help="The maximum number of characters per chunk. Default is -1 which means no chunking.")

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: The input file '{args.input_file}' does not exist.")
        return

    with args.input_file.open("r", encoding="utf-8") as file:
        conversations_data = json.load(file)

    # create_conversations_table(conn)

    write_conversations(
        conversations_data,
        table_name=args.table_name,
        chunk_size = args.chunk_size
    )


if __name__ == "__main__":
    # Connect to the database
    conn = connect_to_db()
    main()
    conn.close()

