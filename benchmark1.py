#!/usr/bin/env python
# coding: utf-8

# # BENCHMARK 1: Compare two embeddings based on Cohere rerank_metric

# In[23]:


from sklearn.metrics import ndcg_score, pairwise
from util import get_embedding_by_id, similarity_search, connect_to_db, print_auto_logged_info, log_args
import os
import cohere
import numpy as np
import argparse
import mlflow 
from pprint import pprint


conn = connect_to_db()

def BM1(bm_ids, emb_col):
    '''
    For example, if we have 10 bm_ids, retrieve 10 most similar docs for each of them,
    and use the similarity score from the top 3 reranked docs, then we will get our average rerank score using 30/100 retrieved documents.
    Further, we will get the cosine similarity score for each of the 100 retrieved docs.
    Lastly, we will use NDCG to see how well our original retrieved docs are ranked compared to the reranked docs.
    '''

    co = cohere.Client(os.getenv("COHERE_API_KEY"))

    all_rerank_scores = []
    all_cosine_sim_scores = []
    all_ndcg_scores = []

    for id in bm_ids:
        emb = get_embedding_by_id(conn, emb_col, id)
        emb_ids, cosine_sim_scores = similarity_search(conn, emb_col, emb, 10) # ignore the first because it's the same doc as emb1
        embs = [get_embedding_by_id(conn, emb_col, id) for id in emb_ids]

        result = co.rerank(query=emb,
                        documents=embs,
                        top_n=10,
                        model='rerank-english-v3.0')
        
        cosine_sim_score = np.mean(cosine_sim_scores)
        rerank_score = np.mean([r.relevance_score for r in result.results][:3]) # we use the top 3 relevance scores after reranking

        ranks = np.array([range(10)]) # original ranks of the retrieved docs
        new_ranks = np.array([[r.index for r in result.results]]) # new ranks after reranking
        ndcg = ndcg_score(new_ranks, ranks, k=3)   

        all_rerank_scores.append(rerank_score)
        all_cosine_sim_scores.append(cosine_sim_score)
        all_ndcg_scores.append(ndcg)

    return {
        'rerank_scores' : all_rerank_scores,
        'ndcg_scores' : all_ndcg_scores,
        'cosine_sim_scores' : all_cosine_sim_scores,
    }
    



# In[24]:


# EMB_COL = 'mpnet_768d'
# TEST_IDS = range(1,11)
# BM1(TEST_IDS, EMB_COL)



# EMB_COL = 'jina_768d'
# TEST_IDS = range(1,11)
# BM1(TEST_IDS, EMB_COL)

# # In[5]:


# EMB_COL = 'doc2vec_20d'
# TEST_IDS = range(1,11)
# BM1(TEST_IDS, EMB_COL) # VERY POOR PERFORMANCE



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run BM1 with specified embedding column and test IDs range.")
    
    # Add arguments for embedding column and test IDs
    parser.add_argument("--emb_col", type=str, required=True, 
                        help="Name of the embedding column to use (e.g., 'mpnet_768d', 'jina_768d', 'doc2vec_20d').")
    parser.add_argument("--test_ids", type=int, nargs=2, required=True, metavar=('START', 'END'),
                        help="Range of test IDs to use in the format: range(START, END).")
    
    args = parser.parse_args()

    log_args(args) # MLFLOW PART 1

    test_ids = range(args.test_ids[0], args.test_ids[1]) 

    mlflow.autolog() # MLFLOW PART 2
    mlflow.set_experiment("BM1")
    mlflow.end_run()
    with mlflow.start_run() as run: # MLFLOW PART 3
        results = BM1(test_ids, args.emb_col)
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    print_auto_logged_info(run.info.run_id)
    pprint(results)
    breakpoint()



    


