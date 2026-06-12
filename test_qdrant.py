import uuid
class Point:
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload

dense_points = [Point(1, 0.9, {"content": "hello"}), Point(2, 0.8, {"content": "world"})]
sparse_points = [Point(2, 10.0, {"content": "world"}), Point(3, 5.0, {"content": "test"})]

def normalize(scores_dict):
    if not scores_dict: return {}
    min_val = min(scores_dict.values())
    max_val = max(scores_dict.values())
    if max_val == min_val: return {k: 0.0 for k in scores_dict}
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}

dense_scores = {str(p.id): p.score for p in dense_points}
sparse_scores = {str(p.id): p.score for p in sparse_points}

norm_dense = normalize(dense_scores)
norm_sparse = normalize(sparse_scores)

all_ids = set(dense_scores.keys()).union(set(sparse_scores.keys()))
fused_scores = {}
for chunk_id in all_ids:
    d_score = norm_dense.get(chunk_id, 0.0)
    s_score = norm_sparse.get(chunk_id, 0.0)
    fused_scores[chunk_id] = (0.6 * d_score) + (0.4 * s_score)

all_points = {str(p.id): p for p in dense_points + sparse_points}
results_unsorted = []
for chunk_id in all_ids:
    p = all_points[chunk_id]
    results_unsorted.append({
        "chunk_id": chunk_id,
        "score": fused_scores[chunk_id],
        "text": p.payload.get("content")
    })

results = sorted(results_unsorted, key=lambda x: x["score"], reverse=True)
print(results)
