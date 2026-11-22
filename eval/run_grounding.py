#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_grounding.py - ground model outputs with the PDC-Bench judge.

Input : results/outputs/{domain}/{model}.json      (written by run_inference_*.py)
Output: results/grounding/{judge}/{domain}/{model}.json
        a dict from sample id to the judged nodes for the three certainty tiers
        (certain_node / inferred_node / stereotype_node), the completion label, the
        judge's raw answer, and its evidence quotes. Re-running resumes.

The judge is called with temperature 0 (GPT-4o, JSON mode) or with prompt caching
(Claude). Each output is judged once, blind to the private value, task, and constraint.

Example (try a few samples first):
    python eval/run_grounding.py --judge gpt-4o --model gpt4o --domain geo --limit 5
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from judge import build_messages, check_answer, map_answer, validate

B = Path(__file__).resolve().parents[1]  # package root
KGF = {d: f'kg/kg_{d}.json' for d in ('geo', 'age', 'race', 'occ')}
JUDGE_DIR = {'gpt-4o': 'gpt4o', 'claude-sonnet-5': 'sonnet5'}
LOCK = threading.Lock()
USAGE = {'in': 0, 'out': 0, 'cache_read': 0, 'calls': 0, 'errors': 0}


def call_judge(judge, msgs, retries=4):
    delay = 2.0
    for att in range(retries):
        try:
            if judge.startswith('gpt'):
                from openai import OpenAI
                cl = OpenAI()
                r = cl.chat.completions.create(model=judge, temperature=0, max_tokens=300,
                                               messages=msgs,
                                               response_format={"type": "json_object"})
                u = r.usage
                with LOCK:
                    USAGE['in'] += u.prompt_tokens; USAGE['out'] += u.completion_tokens
                    cd = getattr(u, 'prompt_tokens_details', None)
                    USAGE['cache_read'] += getattr(cd, 'cached_tokens', 0) or 0
                    USAGE['calls'] += 1
                return r.choices[0].message.content
            else:
                import anthropic
                cl = anthropic.Anthropic()
                sysmsg = [{"type": "text", "text": msgs[0]['content'],
                           "cache_control": {"type": "ephemeral"}}]
                rest = [dict(m) for m in msgs[1:]]
                # Cache the system prompt and all in-context exemplars.
                last_fs = len(rest) - 2  # the message before the output being judged
                if last_fs >= 0:
                    rest[last_fs] = {"role": rest[last_fs]['role'],
                                     "content": [{"type": "text",
                                                  "text": rest[last_fs]['content'],
                                                  "cache_control": {"type": "ephemeral"}}]}
                r = cl.messages.create(model=judge, max_tokens=4000, system=sysmsg,
                                       messages=rest)
                u = r.usage
                with LOCK:
                    USAGE['in'] += u.input_tokens; USAGE['out'] += u.output_tokens
                    USAGE['cache_read'] += getattr(u, 'cache_read_input_tokens', 0) or 0
                    USAGE['calls'] += 1
                for blk in r.content:
                    if getattr(blk, 'type', '') == 'text':
                        txt = blk.text
                        i, j = txt.find('{'), txt.rfind('}')
                        if i >= 0:
                            return txt[i:j + 1]
                raise RuntimeError('no text block')
        except Exception:
            if att == retries - 1:
                with LOCK:
                    USAGE['errors'] += 1
                raise
            time.sleep(delay)
            delay *= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', required=True, help='gpt-4o or claude-sonnet-5')
    ap.add_argument('--model', required=True,
                    help='file name of the evaluated model under results/outputs (e.g., gpt4o, llama_8b)')
    ap.add_argument('--domain', required=True, choices=list(KGF) + ['all'])
    ap.add_argument('--limit', type=int, default=0, help='max samples per domain (0 = all)')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    domains = list(KGF) if a.domain == 'all' else [a.domain]
    KG = {d: json.load(open(B / f)) for d, f in KGF.items()}
    jdir = JUDGE_DIR.get(a.judge, a.judge.replace('/', '_'))

    for d in domains:
        src = B / f'results/outputs/{d}/{a.model}.json'
        rows = {x['id']: x for x in json.load(open(src))}
        outdir = B / f'results/grounding/{jdir}/{d}'
        outdir.mkdir(parents=True, exist_ok=True)
        outp = outdir / f'{a.model}.json'
        done = json.load(open(outp)) if outp.exists() else {}
        todo = [(i, x['secret_node_id'], x['output']) for i, x in rows.items()
                if i not in done and x['secret_node_id'] in KG[d]]
        if a.limit:
            todo = todo[:a.limit]
        print(f'[{d}] {len(rows)} outputs / {len(done)} done / {len(todo)} to judge', flush=True)

        def work(job):
            i, node, out = job
            ids = [node] + KG[d][node].get('ancestors', [])
            chain = [(n, KG[d][n]['level'], KG[d][n]['surface'],
                      KG[d][n].get('aliases', [])) for n in ids]
            if not out.strip():
                rec = dict(certain_node=None, inferred_node=None, stereotype_node=None,
                           completion=None, raw=None, note='empty_output')
                return i, rec
            try:
                ans = json.loads(call_judge(a.judge, build_messages(d, out)))
            except Exception as e:
                return i, dict(error=str(e)[:200])
            bad = check_answer(ans)
            if bad:
                # Treated like a failed call: retried at the end of the run and, if it remains,
                # excluded from every denominator by the metric scripts.
                return i, dict(error='malformed judge answer: ' + '; '.join(bad))
            mapped, notes = map_answer(ans, KG[d], ids)
            rec = dict(mapped)
            rec['completion'] = ans.get('completion')
            rec['raw'] = {t: ans.get(t) for t in ('certain', 'inferred', 'stereotype')}
            rec['evidence'] = ans.get('evidence')
            if notes:
                rec['_notes'] = notes
            rec['_errors'] = validate(rec, chain)
            return i, rec

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for n_done, (i, rec) in enumerate(ex.map(work, todo), 1):
                done[i] = rec
                if n_done % 25 == 0 or n_done == len(todo):
                    with LOCK:
                        json.dump(done, open(outp, 'w'), ensure_ascii=False)
                    print(f'  {n_done}/{len(todo)}  {time.time() - t0:.0f}s  '
                          f'tokens(in/out/cached)={USAGE["in"]}/{USAGE["out"]}/{USAGE["cache_read"]} '
                          f'errors={USAGE["errors"]}', flush=True)
        json.dump(done, open(outp, 'w'), ensure_ascii=False)
        # Retry failed judgments once, sequentially.
        retry = [i for i, r in done.items() if 'error' in r]
        for i in retry:
            x = rows[i]
            done[i] = work((i, x['secret_node_id'], x['output']))[1]
        if retry:
            json.dump(done, open(outp, 'w'), ensure_ascii=False)
            print(f'  retried {len(retry)}; remaining errors: '
                  f'{sum(1 for r in done.values() if "error" in r)}', flush=True)
    print('USAGE:', USAGE, flush=True)


if __name__ == '__main__':
    main()
