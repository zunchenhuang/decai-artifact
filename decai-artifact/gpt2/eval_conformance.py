import argparse, json, math, os, random, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bounds as RS
from conformance_task import ShellTask, load_model, GENERIC_TEXT, WORD

SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2"); ap.add_argument("--ckpt", default=None); ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--tag", required=True); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results/conformance.jsonl"); ap.add_argument("--exp", default="utility,precision,refine,soundness")
    ap.add_argument("--N", type=int, default=4); ap.add_argument("--L_out", type=int, default=6)
    ap.add_argument("--split", default="test"); ap.add_argument("--refine_s", type=int, default=64)
    ap.add_argument("--refine_budget", type=int, default=300); ap.add_argument("--refine_time", type=int, default=1800)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads: torch.set_num_threads(a.threads)
    model, tok = load_model(a.model, a.layers, a.synthetic, a.ckpt)
    V = model.get_output_embeddings().weight.shape[0]
    task = ShellTask(tok, V); mon = task.monitor(); ad = RS.Adapter(model)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = {json.loads(l)["key"] for l in open(a.out)} if os.path.exists(a.out) else set()
    f = open(a.out, "a")
    def emit(rec):
        rec.update(tag=a.tag, model=a.model, ckpt=a.ckpt, layers=len(ad.blocks))
        f.write(json.dumps(rec) + "\n"); f.flush(); print(json.dumps({k: v for k, v in rec.items() if k != "key"}), flush=True)
    exps = a.exp.split(",")
    show = lambda ids: b"".join(task.bytes[i] for i in ids).decode("utf-8", "replace")

    if "utility" in exps and f"{a.tag}|utility" not in done:
        rng = random.Random(1); res = {"word": [0, 0], "risky": [0, 0]}; examples = []
        for kind in ("word", "risky"):
            pool = task.pool[(kind, a.split)]
            for x in rng.sample(pool, min(200 if kind == "word" else 100, len(pool))):
                v = rng.randrange(len(task.verbs)); ids = task.prompt_ids(v, x); y = []
                with torch.no_grad():
                    for _ in range(4):
                        t = int(model(torch.tensor([ids + y])).logits[0, -1].argmax()); y.append(t)
                        if t == task.eos: break
                ok = y == task.target(v, x); res[kind][0] += ok; res[kind][1] += 1
                if len(examples) < 6 or (not ok and len(examples) < 12): examples.append((show(ids), show(y), ok))
        enc = task.encode(GENERIC_TEXT)
        enc = enc[:min(512, getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 512))]
        with torch.no_grad():
            nll = float(model(torch.tensor([enc]), labels=torch.tensor([enc])).loss)
        emit(dict(key=f"{a.tag}|utility", kind="utility", copy_acc=res["word"][0] / max(1, res["word"][1]),
                  refuse_acc=res["risky"][0] / max(1, res["risky"][1]), n_word=res["word"][1], n_risky=res["risky"][1],
                  ppl=math.exp(nll), examples=examples))

    if "precision" in exps:
        for k in (1, 2):
            for s in SIZES:
                ok_any = False
                for i in range(a.N):
                    key = f"{a.tag}|precision|{k}|{s}|{i}"
                    if key in done: ok_any = True; continue
                    rng = random.Random(f"prec{k}{s}{i}")
                    v = rng.randrange(len(task.verbs)); slot = task.sample_slot(rng, s, a.split)
                    spec = task.prompt_sets(v, slot)
                    r = RS.abstract_decode(ad, mon, spec, k, a.L_out)
                    rec = dict(key=key, kind="precision", k=k, s=s, i=i, verdict=r["verdict"], calls=r["calls"],
                               secs=round(r["secs"], 2), width=round(r.get("width", 0.0), 3), n_cand=r.get("n_cand"), step=r.get("step"),
                               n_risky=sum(not WORD.match(task.bytes[x]) for x in slot))
                    if s == 1:
                        y = RS.exact_check(ad, mon, [int(x[0]) for x in spec], k, a.L_out)
                        rec["truth"] = "UNSAFE" if y is not None else "SAFE"
                    emit(rec); ok_any |= r["verdict"] == "VERIFIED"
                if not ok_any and s >= 2: break

    if "refine" in exps:
        for k in (1, 2):
            for i in range(a.N):
                key = f"{a.tag}|refine|{k}|{i}"
                if key in done: continue
                rng = random.Random(f"ref{k}{i}")
                v = rng.randrange(len(task.verbs)); slot = task.sample_slot(rng, a.refine_s, a.split)
                spec = task.prompt_sets(v, slot); pos = len(task.prefix[task.verbs[v][0]])
                r = RS.refine(ad, mon, spec, k, a.L_out, budget=a.refine_budget, time_limit=a.refine_time)
                rec = dict(key=key, kind="refine", k=k, i=i, s=len(slot), verdict=r["verdict"], secs=round(r["secs"], 1),
                           abstract_runs=r["abstract_runs"], exact_checks=r["exact_checks"], proved_leaves=r["proved_leaves"],
                           proved_inputs=r["proved_inputs"])
                if r["witness"]:
                    ids, y = r["witness"]; rec["cex_in"] = show([ids[pos]]); rec["cex_out"] = show(y)
                emit(rec)

    if "soundness" in exps:
        for i in range(12):
            key = f"{a.tag}|soundness|{i}"
            if key in done: continue
            rng = random.Random(f"snd{i}"); k = rng.choice([1, 2]); s = rng.choice([1, 2, 3])
            v = rng.randrange(len(task.verbs)); slot = task.sample_slot(rng, s, a.split, risky_frac=0.5)
            spec = task.prompt_sets(v, slot)
            truth = RS.enumerate_truth(ad, mon, spec, k, 4); r = RS.abstract_decode(ad, mon, spec, k, 4)
            emit(dict(key=key, kind="soundness", k=k, s=s, truth=truth, verdict=r["verdict"],
                      sound=not (r["verdict"] == "VERIFIED" and truth == "UNSAFE")))

    # summary
    R = [json.loads(l) for l in open(a.out)]; R = [r for r in R if r.get("tag") == a.tag]
    print(f"\n== {a.tag}")
    for r in R:
        if r["kind"] == "utility":
            print(f"utility: copy {r['copy_acc']:.1%} (n={r['n_word']}), refusal {r['refuse_acc']:.1%} (n={r['n_risky']}), perplexity {r['ppl']:.1f}")
    for k in (1, 2):
        row = []
        for s in SIZES:
            pr = [r for r in R if r["kind"] == "precision" and r["k"] == k and r["s"] == s]
            if pr: row.append(f"s={s}: {sum(r['verdict'] == 'VERIFIED' for r in pr)}/{len(pr)}"
                              + (f" (alarms at steps {sorted(r['step'] for r in pr if r['verdict'] == 'ALARM')})" if any(r['verdict'] == 'ALARM' for r in pr) else ""))
        print(f"abstract decoding, k={k}: " + ", ".join(row))
    rf = [r for r in R if r["kind"] == "refine"]
    if rf: print("refinement: " + ", ".join(f"k={r['k']} {r['verdict']} ({r['secs']}s, {r['abstract_runs']} runs)" for r in rf))
    sd = [r for r in R if r["kind"] == "soundness"]
    if sd: print(f"soundness: {sum(not r['sound'] for r in sd)} violations / {len(sd)} (unsafe: {sum(r['truth'] == 'UNSAFE' for r in sd)})")

if __name__ == "__main__":
    main()
