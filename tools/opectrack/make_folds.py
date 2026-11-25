"""Split a training info file into two folds by scene (alternating over the sorted scene tokens).

Used for cross-fold sender messages: the sender detector that produces the training
messages of a scene has not been trained on that scene, so training messages are as
imperfect as validation messages.

    python tools/opectrack/make_folds.py INFOS.pkl OUT_DIR
"""
import argparse
import pickle
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infos", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    with args.infos.open("rb") as stream:
        data = pickle.load(stream)
    args.out.mkdir(parents=True, exist_ok=True)
    scenes = sorted({info["scene_token"] for info in data["infos"]})
    fold = {scene: "AB"[index % 2] for index, scene in enumerate(scenes)}
    for name in "AB":
        infos = [info for info in data["infos"] if fold[info["scene_token"]] == name]
        with (args.out / f"infos_{name}.pkl").open("wb") as stream:
            pickle.dump(dict(infos=infos, metadata=data["metadata"]), stream)
        print(f"Fold {name}: {len(infos)} frames")


if __name__ == "__main__":
    main()
