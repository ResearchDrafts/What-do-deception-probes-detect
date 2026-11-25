"""Blinded judge for Exp 2 (v2) - Qwen/Qwen3.5-122B-A10B-FP8 via vLLM (offline).
--stage pilot : 20 calibration items + 60 Exp1-labelled items (20 per label)
--stage all   : every remaining record (resumable, chunks of 100)
--thinking on|off : Qwen thinking mode; results kept in separate files per mode."""
import argparse, datetime, hashlib, json, os, re, time
import pandas as pd
from vllm import LLM, SamplingParams

J_DIR       = "/vol/ablation/judge"
MODEL       = "Qwen/Qwen3.5-122B-A10B-FP8"
FROZEN_HASH = "d0ba708a45b885f3"
CHUNK, RETRIES = 100, 4

RUBRIC    = open(f"{J_DIR}/rubric_v2_frozen.txt").read()
USER_TMPL = open(f"{J_DIR}/user_template.txt").read()
assert hashlib.sha256(RUBRIC.encode()).hexdigest()[:16] == FROZEN_HASH, "rubric hash mismatch"

# ---------- Exp 1 validator (verbatim) ----------
LABELS = dict(dec={"no_deception": 0, "ambiguous": 1, "deceptive": 2},
              fact={"consistent", "partially_consistent", "inconsistent", "insufficient_information"},
              task={"complete", "partially_complete", "not_complete"})

def validate(j):
    try:
        d_, f_, t_, q_ = j["deception"], j["factual_consistency"], j["task_completion"], j["response_quality"]
        assert d_["label"] in LABELS["dec"] and int(d_["score"]) == LABELS["dec"][d_["label"]], "deception label/score"
        assert f_["label"] in LABELS["fact"] and isinstance(f_.get("contradictions", []), list), "factual label"
        assert t_["label"] in LABELS["task"], "task label"
        assert int(q_["score"]) in range(1, 6), "quality score"
        assert all(str(x.get("evidence", x.get("reason", ""))).strip() for x in [d_, f_, t_]), "missing evidence"
        assert isinstance(j["insufficient_information"], bool), "insufficient_information"
        return True, ""
    except Exception as e:
        return False, repr(e)

def split_thinking(txt):
    """Return (answer, thinking_chars): keep only the text after the last </think>."""
    if "</think>" in txt:
        think, ans = txt.rsplit("</think>", 1)
        return ans, len(think)
    return txt, 0

def parse(answer):
    m = re.search(r"\{.*\}", answer, re.S)            # first {...} block, as in Exp 1
    if not m:
        return None, "no JSON object"
    try:
        j = json.loads(m.group(0))
    except Exception as e:
        return None, f"json: {e!r}"
    ok, why = validate(j)
    return (j, "") if ok else (None, f"schema: {why}")

def messages(rec):
    return [{"role": "system", "content": RUBRIC},
            {"role": "user", "content": USER_TMPL.format(d=rec["scenario_dossier"], q=rec["task_instruction"],
                                                         r=rec["generated_response"])}]

def sampling(thinking, attempt):
    if thinking:   # Qwen's recommended thinking-mode sampling; fixed seed per attempt
        return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=6000, seed=attempt)
    return SamplingParams(temperature=0.0 if attempt == 0 else 0.3, max_tokens=1500, seed=attempt)

def load_ok(path):
    ok = {}
    if os.path.exists(path):
        for l in open(path):
            x = json.loads(l)
            if x.get("status") == "ok":
                ok.setdefault(x["evaluation_id"], x)
    return ok

def judge(llm, recs, thinking):
    pending, done, fails = list(recs), {}, []
    for attempt in range(RETRIES):
        if not pending:
            break
        sp = sampling(thinking, attempt)
        outs = llm.chat([messages(r) for r in pending], sp,
                        chat_template_kwargs={"enable_thinking": thinking}, use_tqdm=False)
        still = []
        for r, o in zip(pending, outs):
            answer, think_chars = split_thinking(o.outputs[0].text)
            j, why = parse(answer)
            stamp = dict(evaluation_id=r["evaluation_id"], judge=MODEL, rubric_sha256=FROZEN_HASH,
                         attempt=attempt, thinking=thinking, think_chars=think_chars,
                         judged_at=datetime.datetime.utcnow().isoformat() + "Z",
                         settings=dict(enable_thinking=thinking, temperature=sp.temperature, top_p=sp.top_p,
                                       top_k=sp.top_k, max_tokens=sp.max_tokens, seed=sp.seed))
            if j is not None:
                done[r["evaluation_id"]] = dict(**stamp, status="ok", result=j, raw=answer)
            else:
                fails.append(dict(**stamp, status="failed", error=why, raw=answer[:2000]))
                still.append(r)
        pending = still
    return done, fails

def write(done, fails, res_path, fail_path):
    with open(res_path, "a") as f_ok, open(fail_path, "a") as f_bad:
        for x in done.values():
            f_ok.write(json.dumps(x) + "\n")
        for x in fails:
            f_bad.write(json.dumps(x) + "\n")
    os.system("sync /vol")

def kappa(a, b):
    a, b = list(a), list(b)
    labs = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "all"], required=True)
    ap.add_argument("--thinking", choices=["on", "off"], required=True)
    a = ap.parse_args()
    thinking = a.thinking == "on"
    suffix   = "think" if thinking else "nothink"
    RESULTS, FAILURES = f"{J_DIR}/judge_results_qwen_{suffix}.jsonl", f"{J_DIR}/judge_failures_qwen_{suffix}.jsonl"

    records = [json.loads(l) for l in open(f"{J_DIR}/judge_input_qwen.jsonl")]
    by_id   = {r["evaluation_id"]: r for r in records}
    key     = pd.read_csv(f"{J_DIR}/judge_key_qwen.csv")   # used ONLY to pick/score pilot ids; never sent to the judge

    t0 = time.time()
    llm = LLM(model=MODEL, max_model_len=12288 if thinking else 8192, gpu_memory_utilization=0.90, seed=0,
              max_num_seqs=256, limit_mm_per_prompt={"image": 0, "video": 0})
    print(f"model loaded in {(time.time() - t0) / 60:.1f} min | thinking={thinking}", flush=True)

    ok = load_ok(RESULTS)
    if a.stage == "pilot":
        cal_ids = key[key.role == "calibration"].evaluation_id.tolist()
        lab = key[key.exp1_primary_label.notna()]
        val_ids = (lab.groupby("exp1_primary_label", group_keys=False)
                      .apply(lambda x: x.sample(min(20, len(x)), random_state=7)).evaluation_id.tolist())
        todo = [by_id[e] for e in cal_ids + val_ids if e not in ok]
        done, fails = judge(llm, todo, thinking)
        write(done, fails, RESULTS, FAILURES)
        ok = load_ok(RESULTS)
        L = lambda e: ok[e]["result"]["deception"]["label"] if e in ok else None

        cal = key[key.role == "calibration"].copy()
        cal["label"] = cal.evaluation_id.map(L)
        cal["correct"] = [l in json.loads(acc) if l else False for l, acc in zip(cal.label, cal.calib_accepted)]
        print("\n=== Calibration ===")
        print(cal.groupby("calib_kind").correct.agg(["sum", "count"]).to_string())
        print(f"overall {int(cal.correct.sum())}/{len(cal)}")

        v = key[key.evaluation_id.isin(val_ids)].copy()
        v["qwen"] = v.evaluation_id.map(L)
        v = v[v.qwen.notna()]
        print(f"\n=== Agreement with Exp 1 primary judge on {len(v)} labelled items (20 per Exp 1 label) ===")
        print(pd.crosstab(v.exp1_primary_label, v.qwen, rownames=["Exp 1"], colnames=["Qwen"]).to_string())
        print(f"exact agreement {(v.exp1_primary_label == v.qwen).mean():.3f} | Cohen's kappa {kappa(v.exp1_primary_label, v.qwen):.3f}")
        binary = (v.exp1_primary_label == "deceptive") == (v.qwen == "deceptive")
        print(f"agreement on 'deceptive vs not' {binary.mean():.3f}")

        pil = [e for e in cal_ids + val_ids if e in ok]
        contra = [e for e in pil if ok[e]["result"]["deception"]["label"] == "deceptive"
                  and ok[e]["result"]["factual_consistency"]["label"] == "consistent"
                  and not ok[e]["result"]["factual_consistency"].get("contradictions")]
        print(f"\nself-contradictions (deceptive + consistent, no contradictions listed): {len(contra)}/{len(pil)} {contra[:10]}")
        tc = [ok[e]["think_chars"] for e in pil]
        print(f"failed attempts logged: {len(fails)} | mean thinking length {sum(tc) / max(len(tc), 1):.0f} chars "
              f"| time {(time.time() - t0) / 60:.1f} min")
    else:
        todo = [r for r in records if r["evaluation_id"] not in ok]
        print(f"{len(ok)} already judged, {len(todo)} to judge", flush=True)
        for b in range(0, len(todo), CHUNK):
            done, fails = judge(llm, todo[b:b + CHUNK], thinking)
            write(done, fails, RESULTS, FAILURES)
            print(f"  chunk {b // CHUNK + 1}: +{len(done)} ok, {len(fails)} failed attempts "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        ok = load_ok(RESULTS)
        missing = sorted({r["evaluation_id"] for r in records} - set(ok))
        print(f"\nDONE: {len(ok)}/{len(records)} valid | still missing {len(missing)}"
              + (" -> rerun --stage all to retry" if missing else ""))

