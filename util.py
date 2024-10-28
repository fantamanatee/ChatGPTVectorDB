import psycopg2
import os 
from mlflow import MlflowClient, log_param
import numpy as np

def connect_to_db():
    """
    Connect to a postgres database

    Returns:
        psycopg2 connection: The connection object
    """
    conn = psycopg2.connect(host='127.0.0.1',dbname="testdb", user="postgres", password=os.getenv('POSTGRES_PWD'))
    return conn

def rerank_metric(co, query, docs, top_n=10):
    """
    Computes the sum of relevance scores for the top N documents based on reranking.

    Args:
        co: The cohere client 
        query: The query string used for reranking the documents.
        docs: A list of documents to be reranked.
        top_n (int, optional): The number of top documents to consider for relevance score summation. Defaults to 3.

    Returns:
        float: The ranks and relevance scores for the top N documents.
    """
    results = co.rerank(query=query,
                    documents=docs,
                    top_n=top_n,
                    model='rerank-english-v3.0')
    return np.mean([result.relevance_score for result in results.results])


def similarity_search(conn, emb_name, embedding, n_results=10):
    """
    Perform a similarity search for a given embedding.

    Args:
        conn: The postgres connection.
        emb_name: The name of the column containing the embeddings.
        embedding: The embedding to search for.
        n_results: The number of results to return.

    Returns:
        A list of ids and a list of cosine similarity scores for the top N results.

    Raises:
        Exception: If there is an error executing the query.
    """
    try:
        with conn.cursor() as cur:
            query = f"""
            SELECT id, 1 - ({emb_name} <=> '{embedding}') AS cosine_sim
            FROM conversations
            ORDER BY cosine_sim DESC
            LIMIT {n_results} OFFSET 1;
            """
            cur.execute(query)
            results = cur.fetchall()
    except Exception as e:
        print(e)
    
    ids = [res[0] for res in results]
    cosine_sim_scores = [res[1] for res in results]

    return ids, cosine_sim_scores


def get_embedding_by_id(conn, emb_name, id):
    try:
        with conn.cursor() as cur:
            query = f"SELECT {emb_name} FROM conversations WHERE id = %s"
            cur.execute(query, (id,))
            results = cur.fetchone()
    except Exception as e:
        print(e)
    return results[0]


def log_args(args):
    """
    Logs all arguments from the argparse Namespace as MLflow parameters.

    Parameters
    ----------
    args : Namespace
        The command-line arguments parsed by argparse.
    """
    for key, value in vars(args).items():
        log_param(key, value)
    print("Logged parameters:", vars(args))

def print_auto_logged_info(run_id):
    tags = {k: v for k, v in run_id.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(run_id.info.run_id, "model")]
    print(f"run_id: {run_id.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {run_id.data.params}")
    print(f"metrics: {run_id.data.metrics}")
    print(f"tags: {tags}")

# import json
# import mlflow

# def log_results(results_dict, artifact_name="results.json"):
#     """
#     Logs the results in `results_dict` as MLflow metrics and saves
#     the dictionary as an artifact.

#     Parameters
#     ----------
#     results_dict : dict
#         Dictionary containing metrics to log and save as an artifact.
#     artifact_name : str, optional
#         Name of the artifact file to save (default is 'results.json').
#     """
#     # Log individual metrics in the results dictionary
#     for key, values in results_dict.items():
#         if isinstance(values, (list, tuple)):  # Log each value in lists/tuples as metrics
#             for i, value in enumerate(values):
#                 mlflow.log_metric(f"{key}_{i}", value)
#         else:
#             mlflow.log_metric(key, values)

#     # Save the dictionary as a JSON artifact
#     with open(artifact_name, "w") as f:
#         json.dump(results_dict, f)
#     mlflow.log_artifact(artifact_name)


    
if __name__ == "__main__":
    conn = connect_to_db()
