import bottleneck
import numpy as np
def precision_recall_score(scores, relevance, cutoffs, train_relevance=None,
                           return_all=False, offset=0):
    """
    Compute the recall score at cutoff for binary
    hits = retrieved (intersection) relevant
    total possible = sum(relevant)
    hits / (total possible)
    if train_relevance is passed we do not count it as a hit and ignore it all
    together.
    :param scores: 2-d array of scores
    :param relevance: list[set/list] of relevant indices
    :param cutoffs: int/list of ints, cutoff value
    :param train_relevance: list[set/list] of relevant indices for training
    :param return_all: bool, do not compute the mean return the full array
    :param offset: int, we offset the relevance/user index by this much
    :returns: Recall for each row in the array
    :rtype: np.array
    """
    scores = np.asarray(scores, dtype=np.float32)
    assert scores.ndim == 2
    if not isinstance(cutoffs, list):
        cutoffs = [cutoffs]
    max_cutoff = np.max(cutoffs)
    if max_cutoff > scores.shape[1]:
        raise ValueError("Cutoff must be <= scores got: %s and %s"
                         % (cutoffs, scores.shape))
    # Argsort Indices
    # Partition will ensure up to cutoff is sorted
    recall = np.zeros((len(cutoffs), len(scores)), dtype=np.float32)
    precision = np.zeros((len(cutoffs), len(scores)), dtype=np.float32)
    # indices = np.argsort(-scores, axis=1)
    for K, cutoff in enumerate(cutoffs):
        indices = bottleneck.argpartition(-scores, cutoff-1)[:, :cutoff]
        for user_id, index in enumerate(indices):
            total_count = len(relevance[user_id+offset])
            # Check for hits then divide by total possible
            if total_count > 0:
                if train_relevance is None:
                    hits = np.sum([1.0 for idx in index
                                   if idx in relevance[user_id+offset]])
                else:
                    hits = 0.0
                    count = 0
                    # For each item
                    for idx in index:
                        # Skip if in training set
                        if idx in train_relevance[user_id+offset]:
                            continue
                        if idx in relevance[user_id+offset]:
                            hits += 1.0
                        # Count to exit considering the training relevance
                        count += 1
                        if count == cutoff:
                            break
                precision[K, user_id] = hits / float(cutoff)
                recall[K, user_id] = hits / float(total_count)
    if return_all:
        return precision, recall
    else:
        return precision.mean(axis=1), recall.mean(axis=1)