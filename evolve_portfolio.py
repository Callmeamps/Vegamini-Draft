# evolve_portfolio.py
import json, itertools, hashlib, subprocess, argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--portfolio', default='portfolio.json')
parser.add_argument('--mutations_per_parent', type=int, default=8)
parser.add_argument('--iterations', type=int, default=3)
args = parser.parse_args()

AB_SPACE = {'swarm_size':[16,32,64], 'quality_gate':[0.6,0.65,0.7,0.75], 'stv_margin':[0.05,0.1,0.15,0.2]}

def mutate(parent, k=8):
    base = parent.copy()
    keys, vals = list(AB_SPACE.keys()), list(AB_SPACE.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
    combos.sort(key=lambda c: sum(c[k]!=parent[k] for k in keys)) # prefer small changes
    children = []
    for mut in combos[:k]:
        child = {**base, **mut}
        s = json.dumps({k:child[k] for k in sorted(child) if k!='iteration'}, sort_keys=True)
        child['seed'] = int(hashlib.md5(s.encode()).hexdigest()[:8],16)
        child['cell_id'] = hashlib.md5(s.encode()).hexdigest()[:8]
        child['parent_id'] = parent['cell_id']
        children.append(child)
    return children

def main():
    portfolio_path = Path(args.portfolio)
    if not portfolio_path.exists():
        print(f"Portfolio file {args.portfolio} not found. Run build_doe.py first.")
        return

    parents = [json.loads(Path(f"configs/cell_{cid}.json").read_text()) 
               for cid in json.loads(portfolio_path.read_text())
               if Path(f"configs/cell_{cid}.json").exists()]
    
    if not parents:
        # Try finding by seed if cell_id file doesn't exist
        # This is a bit tricky because cell_id != seed. 
        # In build_doe.py, we saved as cell_{seed}.json.
        # Let's fix that or handle it.
        # Actually, build_doe.py saves as configs/cell_{cfg['seed']}.json
        # But portfolio.json stores cell_id.
        # I should probably save configs with cell_id or mapping.
        print("Parents not found in configs/. checking seeds...")
        parents = []
        for p in Path("configs").glob("cell_*.json"):
            cfg = json.loads(p.read_text())
            if cfg['cell_id'] in json.loads(portfolio_path.read_text()):
                parents.append(cfg)

    for parent in parents:
        for child in mutate(parent, args.mutations_per_parent):
            for it in range(args.iterations):
                child['iteration'] = it
                p = Path(f"configs/mut_{child['seed']}_i{it}.json")
                p.write_text(json.dumps(child))
                subprocess.run(["python", "run.py", "--config", str(p), "--task", "arc", "--file", "sample_tasks.json"])
                subprocess.run(["python", "-m", "vega_mini.sleep", "--config", str(p), "--prune"])

    # Analyze: keep parents + children that beat parent
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
    
    if not rows:
        print("No metrics found.")
        return

    df = pd.DataFrame(rows)
    grp = ['cell_id','parent_id','arch','drop_k','swarm_size','quality_gate']
    # Filter out rows without parent_id (Gen 0) to avoid issues, or fillna
    df['parent_id'] = df.get('parent_id', 'none')
    
    s = df.groupby(grp)['quality'].agg(['mean','std']).reset_index()
    parent_ids = [p['cell_id'] for p in parents]
    parent_mean = df[df['cell_id'].isin(parent_ids)].groupby('cell_id')['quality'].mean().to_dict()
    s['parent_mean'] = s['parent_id'].map(parent_mean)
    s['lift'] = s['mean'] - s['parent_mean']

    # New portfolio: best of each lineage
    new_p = []
    for pid in parent_mean.keys():
        lineage = s[(s['cell_id']==pid) | (s['parent_id']==pid)]
        if not lineage.empty:
            new_p.append(lineage.nlargest(1, 'mean').iloc[0])
            
    if new_p:
        new_df = pd.DataFrame(new_p).nlargest(5, 'mean')
        new_df.to_csv('doe_portfolio_genN.csv', index=False)
        Path('portfolio.json').write_text(json.dumps(new_df['cell_id'].tolist()))
        print(f"New portfolio: {len(new_df)} variants")
        print(new_df[['mean','lift','arch','drop_k','cell_id']])
    else:
        print("No new variants found to update portfolio.")

if __name__ == "__main__":
    main()
