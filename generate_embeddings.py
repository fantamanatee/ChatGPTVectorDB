#!/usr/bin/env python
# coding: utf-8

# In[3]:


from util import connect_to_db
from sentence_transformers import SentenceTransformer
from psycopg2 import sql
from tqdm import tqdm
import argparse

# In[4]:




def make_embedding_column_name(model_name, ndim):
    """ Make an n-dimensional embedding column name using the model name and the number of dimensions
    """
    return f"{model_name}_{ndim}d"
    

def make_embedding_column(conn, table_name, model_name, ndim):
    """ create an n-dimensional embedding column in the conversations table using the connection. 
    """

    emb_col_name = make_embedding_column_name(model_name, ndim)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} vector({})").format(
                sql.Identifier(table_name),
                sql.Identifier(emb_col_name),
                sql.Literal(ndim)
            )
        )
        print(emb_col_name)
        print(cur.statusmessage)
    conn.commit()
    
    

def generate_embeddings(conn, model, model_name, table_name, ndim):
    ''' For every row in the table, generate an n-dimensional embedding and store it in the appropriate column
    '''
    emb_col_name = make_embedding_column_name(model_name, ndim)

    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT id, messages FROM {table}").format(
            table = sql.Identifier(table_name)
        ))
        rows = cur.fetchall()

        for id, text in tqdm(rows):
            embedding = model.encode(text)
            query = sql.SQL("UPDATE {table} SET {column} = %s WHERE id = %s").format(
                column = sql.Identifier(emb_col_name),
                table = sql.Identifier(table_name),
            )
            cur.execute(
                query, (embedding.tolist(), id)
            )
            conn.commit()



def main():
    # Setup argparse
    parser = argparse.ArgumentParser(description="Generate embeddings for a specified table using a specified model.")
    parser.add_argument("--table_name", type=str, required=True, help="Name of the table to update with embeddings.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (used for embedding column).")
    parser.add_argument("--model_path", type=str, required=True, help="Path or name of the model for SentenceTransformer.")
    parser.add_argument("--ndim", type=int, default=768, help="Dimension of the embedding vector.")
    args = parser.parse_args()

    conn = connect_to_db()
    make_embedding_column(conn, args.table_name, args.model_name, args.ndim)
    model = SentenceTransformer(args.model_path, trust_remote_code=True)
    generate_embeddings(conn, model, args.model_name, args.table_name, args.ndim)
    conn.close()

if __name__ == "__main__":
    main()