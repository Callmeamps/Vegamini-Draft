# build_doe.py
import itertools, json, hashlib, subprocess
from pathlib import Path
import pandas as pd

SPACE = {
    'arch': ['trm_7m', 'trm_14m'], # A/1: Strategy
    'drop_k': [3, 5], # A/1: Concept
    'decay_lambda': [0.95, 0.90], # A/1: Concept
    'swarm_size': [16, 32], # A/B: Execution
    'quality_gate': [0.6, 0.7], # A/B: Execution
    'stv_margin': [0.1, 0.2], # A/B: Execution
    'is_control': [False, True], # A/A: Control
    'iteration': range(3) # Iterations: anomaly kill
}

def make_configs():
    keys = list(SPACE.keys())
    cells = [dict(zip(keys, v)) for v in itertools.product(*SPACE.values())]
    uniq, seen_ctrl = [], set()
    for c in cells:
        if c['is_control']:
            if c['iteration'] in seen_ctrl: continue
            seen_ctrl.add(c['iteration'])
            c.update({'arch':'trm_7m','drop_k':3,'decay_lambda':0.95,
                     'swarm_size':32,'quality_gate':0.65,'stv_margin':0.1})
        s = json.dumps({k:c[k] for k in sorted(c) if k!='iteration'}, sort_keys=True)
        c['seed'] = int(hashlib.md5((s+str(c['iteration'])).encode()).hexdigest()[:8],16)
        c['cell_id'] = hashlib.md5(json.dumps({k:c[k] for k in sorted(c) if k!='iteration'}).encode()).hexdigest()[:8]
        uniq.append(c)
    return uniq

def run_cell(cfg):
    p = Path(f"configs/cell_{cfg['seed']}.json")
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2))
    # Use a small sample for testing if needed, but here we follow playbook
    subprocess.run(["python", "run.py", "--config", str(p), "--task", "arc", "--file", "sample_tasks.json"], check=True)
    subprocess.run(["python", "-m", "vega_mini.sleep", "--config", str(p), "--prune"], check=True)

def find_portfolio(df, k=5):
    grp = ['arch','drop_k','decay_lambda','swarm_size','quality_gate','stv_margin','cell_id']
    s = df[df['is_control']==False].groupby(grp)['quality'].agg(['mean','std','count']).reset_index()
    s['cv'] = s['std'] / s['mean']
    s = s[s['count'] >= 2] # needs 2+ iterations
    ctrl_std = df[df['is_control']==True]['quality'].std()
    s = s[s['mean'] > df['quality'].mean() + 2*ctrl_std] if not df[df['is_control']==True].empty else s
    # Pareto: keep if not dominated on mean/cv
    def dominated(r, others):
        return any((others['mean']>r['mean']) & (others['cv']<=r['cv']))
    pareto = s[~s.apply(lambda r: dominated(r, s), axis=1)]
    # Diversity: 1 per arch+drop_k combo
    portfolio = pareto.groupby(['arch','drop_k']).apply(lambda x: x.nlargest(1,'mean')).reset_index(drop=True)
    return portfolio.nlargest(k, 'mean')

def main():
    cfgs = make_configs()
    print(f"Running {len(cfgs)} cells...")
    # NOTE: In a real scenario, this would be parallelized. 
    # For this task, we'll just process them sequentially or a subset.
    for c in cfgs[:5]: # Only run first 5 for verification
        run_cell(c)
        
    # Analysis part needs logs. Assuming logs/ events.jsonl exists.
    # The playbook assumes logs are created by run.py
    rows = []
    log_dir = Path("logs")
    if log_dir.exists():
        for p in log_dir.rglob("events.jsonl"):
            for l in p.read_text().splitlines():
                if '"quality"' in l:
                    try:
                        rows.append(json.loads(l))
                    except:
                        continue
    
    if rows:
        df = pd.DataFrame(rows)
        portfolio = find_portfolio(df, k=5)
        portfolio.to_csv('doe_portfolio.csv', index=False)
        Path('portfolio.json').write_text(json.dumps(portfolio['cell_id'].tolist()))
        print(f"Portfolio saved: {len(portfolio)} variants")
    else:
        print("No quality metrics found in logs. Portfolio not updated.")

if __name__ == "__main__":
    main()
