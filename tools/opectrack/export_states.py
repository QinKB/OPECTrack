"""State posteriors of a sequential inference run -> the cache used to start training clips (``state_cache``).

    python tools/opectrack/export_states.py RESULTS.pkl INFOS.pkl OUT.pt

RESULTS.pkl: output of tools/test.py for a model with state="evidence", run over the frames of INFOS.pkl in
sequence order (the model writes its posterior into every frame's ``sync`` entry). The cache holds, per frame
token, the posterior after that frame and its time, and the token of the frame before it in its sequence.
"""
import argparse
import pickle

from pathlib import Path

import torch

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("infos", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    with args.results.open("rb") as stream:
        results = pickle.load(stream)
    if isinstance(results, dict) and "bbox_results" in results:
        results = results["bbox_results"]
    with args.infos.open("rb") as stream:
        infos = sorted(pickle.load(stream)["infos"], key=lambda item: (item["scene_token"], item["timestamp"]))

    previous, last = {}, None
    for info in infos:
        token = info["token"]
        previous[token] = last["token"] if last is not None and last["scene_token"] == info["scene_token"] else None
        last = info
    state = {entry["sync"]["token"]: dict(mean=entry["sync"]["state_mean"], cov=entry["sync"]["state_cov"],
                                          time=entry["sync"]["time"], memory=entry["sync"].get("memory"))
             for entry in results if "sync" in entry and "state_mean" in entry["sync"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=state, previous=previous, source=str(args.results)), args.out)
    print(f"Exported {len(state)} frame states to {args.out}")


if __name__ == "__main__":
    main()
