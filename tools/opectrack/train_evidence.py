"""Pretrain the point-evidence head from cached LiDAR crops."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "projects" / "mmdet3d_plugin"))

from opectrack.core import evidence as ev  # noqa: E402
from opectrack.core.pipeline import to_receiver  # noqa: E402
from opectrack.data import protocol  # noqa: E402

def instances(data, cache, inf_items):
    """Flat tensors of every sender instance of the cached frames."""
    out = dict(points=[], count=[], truth=[], xy=[], yaw=[], size=[])
    for seq in sorted(data):
        for fr in data[seq]:
            rec = cache.get(fr["vehicle_frame"])
            if rec is None:
                continue
            rem = to_receiver(inf_items[fr["infrastructure_frame"]],
                              torch.as_tensor(fr["i2v"], dtype=torch.float32),
                              float(fr["time_v"] - fr["time_i"]))
            assert rem["xy"].shape[0] == rec["count"].shape[0]
            out["points"].append(rec["points"])
            out["count"].append(rec["count"])
            out["truth"].append(rec["truth"])
            out["xy"].append(rem["xy"])
            out["yaw"].append(rem["yaw"])
            out["size"].append(rem["size"])
    return {k: torch.cat(v) for k, v in out.items()}


def make_batch(inst, index, gen, device, train=True, negative=0.2, drop=0.5):
    points = inst["points"][index].to(device).float()
    count = inst["count"][index].to(device)
    valid = torch.arange(points.shape[1], device=device)[None, :] < count[:, None]
    truth = inst["truth"][index].to(device)
    xy, yaw, size = (inst[k][index].to(device) for k in ("xy", "yaw", "size"))
    n = len(index)
    rnd = lambda *s: torch.rand(*s, generator=gen, device=device)
    if train:
        target_v = (rnd(n, 2) * 2 - 1) * 2.5                     # wanted displacement, receiver frame
        neg = rnd(n) < negative
        ang = rnd(n) * 2 * math.pi
        far = (6.0 + rnd(n)) [:, None] * torch.stack((ang.cos(), ang.sin()), -1)
        target_v = torch.where(neg[:, None], far, target_v)
        shift = truth - target_v
        heading = yaw + torch.randn(n, generator=gen, device=device) * math.radians(3.0)
        valid = valid & (rnd(*valid.shape) >= rnd(n, 1) * drop)
    else:
        target_v, neg, shift, heading = truth, torch.zeros(n, dtype=torch.bool, device=device), \
            torch.zeros(n, 2, device=device), yaw
    c, s = heading.cos(), heading.sin()
    target = torch.stack((c * target_v[:, 0] + s * target_v[:, 1],
                          -s * target_v[:, 0] + c * target_v[:, 1]), -1)
    feats, mask, glob, cnt = ev.canonical(points, valid, shift, heading, size, xy + shift, generator=gen)
    return feats, mask, glob, cnt, target, ~neg


@torch.no_grad()
def head_metrics(head, inst, device, gen):
    """Displacement error of the head at the recorded position (the real offset), val."""
    head.eval()
    idx = torch.nonzero(torch.isfinite(inst["truth"][:, 0])).reshape(-1)
    err, cnt, val, sig = [], [], [], []
    for k in range(0, len(idx), 2048):
        feats, mask, glob, c, target, _ = make_batch(inst, idx[k:k + 2048], gen, device, train=False)
        out = head(feats, mask, glob)
        err.append((out["mu"] - target).norm(dim=-1).cpu())
        cnt.append(c.cpu())
        val.append(out["valid"].sigmoid().cpu())
        sig.append(torch.linalg.inv(out["info"]).diagonal(dim1=1, dim2=2).sum(-1).sqrt().cpu())
    err, cnt, val, sig = map(torch.cat, (err, cnt, val, sig))
    none = inst["truth"][idx].norm(dim=-1)
    rows = {}
    for name, m in (("all", cnt >= 0), ("points >= 20", cnt >= 20), ("points 5-19", (cnt >= 5) & (cnt < 20)),
                    ("points < 5", cnt < 5)):
        if m.any():
            rows[name] = dict(n=int(m.sum()), median=float(err[m].median()),
                              rmse=float((err[m] ** 2).mean().sqrt()),
                              none_rmse=float((none[m] ** 2).mean().sqrt()),
                              valid=float(val[m].mean()), sigma=float(sig[m].median()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="holds cache-train.pt / cache-val.pt; results are written here")
    ap.add_argument("--items-dir", required=True)
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run = Path(args.run_dir)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    pairs = {s: torch.load(Path(args.pairs_dir) / f"spd_pairs_{s}.pt") for s in ("train", "val")}
    items = {s: protocol.load_items(Path(args.items_dir) / f"items-inf-{s}.pt", args.score)
             for s in ("train", "val")}
    cache = {s: torch.load(run / f"cache-{s}.pt")["frames"] for s in ("train", "val")}
    val_inst = instances(pairs["val"], cache["val"], items["val"])
    head = ev.EvidenceHead().to(device)
    log = []
    inst = instances(pairs["train"], cache["train"], items["train"])
    pos = torch.nonzero(torch.isfinite(inst["truth"][:, 0]) & (inst["count"] >= 1)).reshape(-1)
    print(f"train: {inst['count'].shape[0]} sender instances, {len(pos)} supervised; "
          f"val: {val_inst['count'].shape[0]} instances; parameters {sum(p.numel() for p in head.parameters())}", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(pos) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    cpu_gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        head.train()
        agg, nb = dict(nll=0.0, l1=0.0, bce=0.0), 0
        perm = pos[torch.randperm(len(pos), generator=cpu_gen)]
        for k in range(0, len(perm), args.batch):
            feats, mask, glob, _, target, positive = make_batch(inst, perm[k:k + args.batch], gen, device)
            loss, parts = ev.evidence_loss(head(feats, mask, glob), target, positive)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step()
            sched.step()
            for key in agg:
                agg[key] += parts[key]
            nb += 1
        row = dict(epoch=epoch + 1, **{k: v / nb for k, v in agg.items()})
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            row["val"] = head_metrics(head, val_inst, device, gen)["all"]
        log.append(row)
        print(json.dumps(row), flush=True)
    torch.save(dict(head=head.state_dict(), args=vars(args), log=log), run / f"head-s{args.seed}.pt")


if __name__ == "__main__":
    main()
