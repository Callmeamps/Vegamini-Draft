# VegaMini v0.1 Architecture

[Moved from README.md]

---

## Layer 0: Input Anchors
- `z_0 ~ N(0,I)`, `a_input = Embed(x)`
- Brightness `b_input` starts high, decays as `Σ b_learned` grows.

## Layer 1: Dynamic Lighthouses
- FAISS index + SQLite metadata
- Each lighthouse: `(id, vec, b, q, y_context, task_id, birth, last_reinforce)`

## Layer 2: Controller
- 7M TRM from the paper, with flow matching inside the `z` update
- Outer loop = TRM recursion; Inner loop = ODE solve with anchors

## Layer 3: Sleep
- Nightly job: replay, dream, nightmare, prune, merge
- Updates only `b_i`, never weights for v0.1

## File Structure
```
vega_mini/
├── controller/
│   ├── trm.py # 7M TRM, modified z-update
│   └── flow.py # anchored ODE solver
├── memory/
│   ├── punk.py # FAISS + SQLite wrapper
│   └── lighthouse.py # drop/reinforce/decay logic
├── sleep/
│   ├── dream.py # generative replay
│   ├── nightmare.py # adversarial probe
│   └── consolidate.py # replay + prune + merge
├── eval/
│   └── quality.py # Q_φ score
├── logging/
│   └── events.py # Structured JSONL/CSV logging
├── vis/
│   └── dashboard.py # Plotly visualization tools
├── run.py # day loop: swarm + STV + write
└── sleep.py # night loop orchestrator
```

## Layer 4: Observability
- **Event Logs**: JSONL file with timestamped system events.
- **Metrics**: CSV for numeric time-series tracking (quality, margins).
- **Visualization**: Interactive PCA scatter plots for latent space analysis.

## Core Contracts

### 1. Anchored Flow Solve
```python
# controller/flow.py
def solve_flow(z0, x, y, anchors, t_steps=6):
    z = z0
    dt = 1.0 / t_steps
    for t in range(t_steps):
        v = model_v(z, t*dt, x, y) # your velocity net
        # lighthouse pull
        pull = torch.zeros_like(z)
        for a in anchors: # a = (vec, b, y_ctx, task_id)
            if sim(y, a.y_ctx) < 0.5: continue # conditional
            k = a.b * torch.exp(-||z - a.vec||**2 / sigma**2)
            pull += k * (a.vec - z)
        v_total = v + pull
        z = z + v_total * dt
    return z
```

### 2. Lighthouse schema
```sql
CREATE TABLE lighthouses (
  id INTEGER PRIMARY KEY,
  vec BLOB, -- 1024-dim fp16
  b REAL, -- brightness
  q REAL, -- quality at birth
  y_context TEXT, -- hash of y
  task_id TEXT,
  birth REAL, -- timestamp
  last_reinforce REAL
);
CREATE INDEX idx_vec ON lighthouses(vec); -- FAISS handles this
```

### 3. Day loop: swarm + STV + write
```python
# run.py
def day_step(x, task_id):
    y_candidates, z_trajs = [], []
    anchors = punk.get_live_anchors(task_id, top_k=64)
    # swarm
    for _ in range(32):
        z0 = torch.randn(1024)
        z_final = solve_flow(z0, x, y=None, anchors=anchors)
        y = trm.update_y(z_final, x)
        y_candidates.append(y)
        z_trajs.append(z_final)
    # STV on y clusters
    clusters = cluster_y(y_candidates)
    ballots = rank_by_worker(clusters)
    y_win, margin = stv(ballots)
    # quality gate
    q = quality_model(z_trajs[win_idx], y_win, x)
    if q < 0.65: return y_win, None
    # drop lighthouses: pick low-energy points on winning path
    if q > 0.8:
        traj = z_trajs[win_idx_path]
        stable_pts = traj[energy < median(energy)]
        for pt in stable_pts[:3]:
            punk.drop_lighthouse(pt, b=1.0, q=q, y=y_win, task_id=task_id)
    # reinforce any anchors we passed near
    punk.reinforce_nearby(traj, delta_b=0.1*q)
    return y_win, q
```

### 4. Sleep loop
```python
# sleep.py
def night_cycle():
    # 1. Replay: sample 2k lighthouses weighted by b*q
    for a in punk.sample_live(k=2000, weight='b*q'):
        z = solve_flow(a.vec, x=None, y=a.y_context, anchors=[])
        loss = recon_loss(z, a.vec)
        if loss < tau: punk.reinforce(a.id, 0.05)
        else: punk.decay(a.id, 0.2)
    # 2. Dreams: generative test
    for _ in range(500):
        a = punk.sample_live(k=1)[0]
        x_dream = decode(a.vec + 0.1*randn())
        z = solve_flow(randn(), x_dream, y=None, anchors=punk.get_live())
        q = quality_model(z, y_pred, x_dream)
        if q > 0.7: punk.reinforce(a.id, 0.02)
    # 3. Nightmares: adversarial
    for _ in range(200):
        x_night = adversary.generate(a.vec)
        z = solve_flow(randn(), x_night, y=None, anchors=punk.get_live())
        if fail(z): punk.slash(a.id, 0.5); punk.drop_reef(z)
    # 4. Prune + merge
    punk.decay_all(lambda_=0.95)
    punk.delete_where('b < 0.05')
    punk.merge_nearby(delta=0.1)
```

### 5. Quality model `Q_φ`
Tiny 3-layer MLP. Input: `[z, embed(y), embed(x), stv_margin]`. Output: scalar.
Train it online: when STV margin >0.9 and user thumbs-up, label=1. When margin <0.2 or fail, label=0. Retrain nightly on 10k pairs.

## Dependencies
```
torch>=2.2
faiss-cpu # or faiss-gpu
einops
numpy
sqlite3 # stdlib
```

## Boot sequence
1. `python init_db.py` — creates SQLite + empty FAISS.
2. `python train_quality.py --bootstrap 1000` — cold-start Q_φ using synthetic data.
3. `python run.py --task arc --file train.json` — day loop. Writes lighthouses.
4. `python sleep.py` — runs after 1k queries or 24h.
5. Repeat. After 10k queries you’ve got a charted latent space.

## What we’re explicitly NOT building in v0.1
- Leagues, STV Council, promotion — that’s external.
- Spatial substrate, gravity, satellites — no topology.
- Gene transfer, raids — single World only.
- Weight updates — sleep only touches `b_i`. TRM stays frozen so you can measure if lighthouses help first.

---

This is buildable in ~1.5k lines. The only research risk is the `solve_flow` step: does adding lighthouses actually reduce recursion depth from 6→2? You’ll know in 2 days of logs.
