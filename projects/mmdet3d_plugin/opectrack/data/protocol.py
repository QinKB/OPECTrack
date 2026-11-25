"""Detection loading and match targets for training."""
import torch


def assign_to_labels(det_xy, label_xy, max_dist):
    """One-to-one assignment of detections to annotated boxes by centre distance.

    Greedy over pairs in order of distance (closest first). Returns, for every label,
    the index of its detection or -1.
    """
    out = torch.full((label_xy.shape[0],), -1, dtype=torch.long)
    if det_xy.shape[0] == 0 or label_xy.shape[0] == 0:
        return out
    dist = (label_xy[:, None, :] - det_xy[None, :, :]).norm(dim=-1)
    order = torch.argsort(dist.flatten())
    used = torch.zeros(det_xy.shape[0], dtype=torch.bool)
    for flat in order.tolist():
        g, d = divmod(flat, det_xy.shape[0])
        if dist[g, d] > max_dist:
            break
        if out[g] < 0 and not used[d]:
            out[g] = d
            used[d] = True
    return out


def load_items(path, min_score=0.0):
    """Detections written by ``tools/opectrack/export_items.py``, score-filtered."""
    items = torch.load(path, map_location="cpu")["items"]
    return {token: {k: v[boxes["score"] >= min_score] for k, v in boxes.items()}
            for token, boxes in items.items()}
