import os, sys, torch, transformers as T, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bounds as RS
torch.manual_seed(0)
V = 300
CFGS = {
 "gpt2 (gelu_new)": T.GPT2Config(n_layer=3, n_embd=64, n_head=4, vocab_size=V, bos_token_id=0, eos_token_id=0),
 "gpt2 (gelu)":     T.GPT2Config(n_layer=3, n_embd=64, n_head=4, vocab_size=V, bos_token_id=0, eos_token_id=0, activation_function="gelu"),
 "gpt2 (relu)":     T.GPT2Config(n_layer=3, n_embd=64, n_head=4, vocab_size=V, bos_token_id=0, eos_token_id=0, activation_function="relu"),
}
def main():
    ok = True
    for fam, cfg in CFGS.items():
        m = T.AutoModelForCausalLM.from_config(cfg).float().eval()
        with torch.no_grad():                         # larger weights than the default init, so that bounds are stressed
            for p in m.parameters(): p.mul_(4.0) if p.dim() > 1 else None
        ad = RS.make_adapter(m)
        # (1) concrete equivalence
        err = 0.0; rmax = 0.0
        for _ in range(5):
            ids = torch.randint(2, V, (int(torch.randint(3, 20, (1,))),)).tolist()
            c, r = ad.final_box([torch.tensor([i]) for i in ids])
            z = ad.G @ c; zh = ad.logits(ids)
            err = max(err, float((z - zh).abs().max())); rmax = max(rmax, float(r.abs().max()))
        # (2) containment of logits for random sets
        viol = 0.0
        for _ in range(5):
            T_ = 10; sets = [torch.randint(2, V, (int(torch.randint(1, 6, (1,))),)) for _ in range(T_)]
            c, r = ad.final_box(sets); zc = ad.G @ c; zr = ad.G.abs() @ r
            for _ in range(20):
                ids = [int(s[torch.randint(len(s), (1,))]) for s in sets]
                zh = ad.logits(ids)
                viol = max(viol, float((zh - (zc + zr)).max()), float(((zc - zr) - zh).max()))
        # (3) containment with a position of 100 tokens (evaluated as one box row, ROWS_MAX = 64)
        for _ in range(2):
            sets = [torch.randint(2, V, (1,)) for _ in range(4)] + [torch.randperm(V - 2)[:100] + 2] + [torch.randint(2, V, (1,)) for _ in range(3)]
            c, r = ad.final_box(sets); zc = ad.G @ c; zr = ad.G.abs() @ r
            for _ in range(20):
                ids = [int(s[torch.randint(len(s), (1,))]) for s in sets]
                zh = ad.logits(ids)
                viol = max(viol, float((zh - (zc + zr)).max()), float(((zc - zr) - zh).max()))
        good = err < 1e-3 and viol <= 1e-4
        ok &= good
        print(f"{fam:16s} concrete max|z - z_hf| = {err:.2e}  radius {rmax:.1e}   containment max violation = {viol:.2e}   {'OK' if good else 'FAIL'}")
    print("ALL OK" if ok else "FAILURES")

if __name__ == "__main__":
    main()
