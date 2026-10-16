"""Loss ablation: conformance loss with mean only vs. mean + worst-token term (slug, seed 0, L=8)."""
import json, os, warnings, torch; warnings.filterwarnings("ignore")
torch.set_num_threads(int(__import__("os").environ.get("THREADS", "1")))
from tasks import Monitor; from analysis import Analyzer, full_ascii_spec; from run_sweep import load; from battery import BATTERY
import train_conf
if not os.path.exists("models/slug_conffocus_s0.pt"):
    train_conf.main("slug", 4000, 0, 50, focus=True)
res = {}
for name in ("slug_conf_s0", "slug_conffocus_s0"):
    lm = load(name); ver = tot = 0
    for prop, fb in BATTERY["slug"]["P"].items():
        for k in (1, 2, 5, 10, 50):
            tot += 1; ver += Analyzer(lm, Monitor(fb), k, backend="tibp").run(full_ascii_spec(8), 8, mode="join", time_limit=60)["verdict"] == "VERIFIED"
    acc = torch.load(f"models/{name}.pt", weights_only=False)["acc"]
    res[name] = dict(verified=ver, instances=tot, acc={str(k): v for k, v in acc.items()})
json.dump(res, open("results/loss_ablation.json", "w"), indent=1); print(res)
