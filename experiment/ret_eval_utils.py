import csv
import numpy as np


def dcg(relevances, k):
    """Discounted Cumulative Gain at k."""
    relevances = np.asfarray(relevances)[:k]
    if relevances.size:
        return relevances[0] + np.sum(relevances[1:] / np.log2(np.arange(2, relevances.size + 1)))
    return 0.


def ndcg(rankings, correct_docs, corpus_ids, k=10):
    """Normalized Discounted Cumulative Gain at k."""
    relevances = [1 if doc_id in correct_docs else 0 for doc_id in corpus_ids]
    sorted_relevances = [relevances[idx] for idx in rankings[:k]]
    ideal_relevance = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal_relevance, k)
    actual_dcg = dcg(sorted_relevances, k)
    if ideal_dcg == 0:
        return 0.
    return actual_dcg / ideal_dcg


def evaluate_retrieval(rankings, correct_docs, corpus_ids, k=10):
    recalled_docs = set(corpus_ids[idx] for idx in rankings[:k])
    recall = float(any(doc in recalled_docs for doc in correct_docs))
    ndcg_score = ndcg(rankings, correct_docs, corpus_ids, k)
    return recall, ndcg_score


def evaluate_retrieval_turn2session(rankings, correct_docs, corpus_ids, k=10):
    # convert turn-level labels/results into session-level and then evaluate
    def strip_turn_id(docid):
        return docid.split('-')[0]
    correct_docs = list(set([strip_turn_id(x) for x in correct_docs]))

    # revise k to handle document-level retrieval
    corpus_ids = [strip_turn_id(x) for x in corpus_ids]
    effective_k = k
    unique_docids = set(corpus_ids[idx] for idx in rankings[:effective_k])
    while effective_k <= len(corpus_ids) and len(unique_docids) < k:
        effective_k += 1
        unique_docids = set(corpus_ids[idx] for idx in rankings[:effective_k])

    return evaluate_retrieval(rankings, correct_docs, corpus_ids, k=effective_k)

def retrieval_csv_save(csv_path, retrieval_res):
    k_values = [1, 3, 5, 10]
    header = ["Metric"] + [f"@{k}" for k in k_values]

    with open(csv_path, mode='w', newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for metric in ["recall", "ndcg"]:
            row = [ metric]
            for k in k_values:
                key = f"{metric}@{k}"
                value = sum(retrieval_res[key]) / len(retrieval_res[key])
                row.append(value)
            writer.writerow(row)

