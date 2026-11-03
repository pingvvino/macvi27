# LOOKOUT VO -- learned speed SHAPE from the water flow, all cameras. Input per 8-Hz step: the vertical-flow profile on 24 angles below the horizon / v^2 / the clip's median water
# (asinh), fit quality, pitch / yaw rate, horizon change, finite share + 6 camera channels (wide, flir-VIS, zoom, sionyx, flir-IR, infrared).
# Target: log GPS speed minus its clip median. Dilated temporal CNN, 320 channels x 8 blocks (~6.6 M parameters), GroupNorm, dropout 0.2,
# AdamW wd 0.05, cosine lr, mixed precision. Augmentation: synthetic speed change on a random segment (profile x f and target + log f, as
# the water law), time stretch 0.8-1.25, noise, profile-band dropout. 5 folds by clip over TRAIN (all cameras) -> out-of-fold predictions;
# val / test = the mean of the 5 fold models; checkpoints every 10 epochs. Variants: allcam (camera channels) and allcam_nocam.
import os, glob, json, time, math, numpy as np, torch, torch.nn as nn
SEED = int(os.environ.get('FM_SEED', '0')); REFIT = False   # FM_SEED 0..3; the folds use the training clips only (no training on val or test)
torch.manual_seed(SEED); np.random.seed(SEED)
DDIR = next(p for p in glob.glob('/kaggle/input/**/clip_*.npz', recursive=True)).rsplit('/', 1)[0] if 'FM_DATA' not in os.environ else os.environ['FM_DATA'] + '/allcam'   # Kaggle flattens a single-folder dataset; FM_DATA wins (Colab also has /kaggle/input)
OUT = os.environ['FM_OUT'] if 'FM_OUT' in os.environ else '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; AMP = DEV == 'cuda'; print('device', DEV, '| data', DDIR, flush=True)
CROP, BATCH = 480, 32; STEPS = int(os.environ.get('FM_STEPS', '100')); EPOCHS = int(os.environ.get('FM_EPOCHS', '60')); CK = int(os.environ.get('FM_CK', '10'))
HID = int(os.environ.get('FM_HID', '320')); NQ = 24
class Block(nn.Module):
    def __init__(s, h, k, d, p):
        super().__init__(); s.n = nn.GroupNorm(1, h); s.c1 = nn.Conv1d(h, h, k, padding=d * (k - 1) // 2, dilation=d); s.a = nn.GELU(); s.dr = nn.Dropout(p); s.c2 = nn.Conv1d(h, h, 1)
    def forward(s, x): return x + s.c2(s.dr(s.a(s.c1(s.n(x)))))
class TCN(nn.Module):
    def __init__(s, ci, h=HID, k=7, dil=(1, 2, 4, 8, 16, 32, 1, 4), p=0.2):
        super().__init__(); s.inp = nn.Conv1d(ci, h, 1); s.blocks = nn.Sequential(*[Block(h, k, d, p) for d in dil]); s.out = nn.Sequential(nn.GroupNorm(1, h), nn.Conv1d(h, 1, 1))
    def forward(s, x): return s.out(s.blocks(s.inp(x)))[:, 0]
for VAR in os.environ.get('FM_VARS', 'allcam,allcam_nocam').split(','):
    NOCAM = VAR.endswith('_nocam'); DATA = DDIR; TAG = f'{VAR}_s{SEED}' + ('_refit' if REFIT else ''); print('=== variant', TAG, flush=True)
    C = {}
    for f in sorted(glob.glob(f'{DATA}/clip_*.npz')):
        z = np.load(f); X = z['X'][:, :-6] if NOCAM else z['X']; C[os.path.basename(f)[:-4]] = dict(X=X.astype(np.float32), y=z['y'].astype(np.float32), split=str(z['split']))
    tr = [k for k, c in C.items() if c['split'] == 'train']; va = [k for k, c in C.items() if c['split'] == 'val']; te = [k for k, c in C.items() if c['split'] == 'test']
    allX = np.concatenate([C[k]['X'] for k in tr]); mu = np.nanmean(allX, 0); sd = np.nanstd(allX, 0) + 1e-6
    if not NOCAM: mu[-6:], sd[-6:] = 0.0, 1.0                    # one-hot stays 0 / 1
    for c in C.values(): c['Xn'] = np.nan_to_num((c['X'] - mu) / sd).astype(np.float32)
    CI = C[tr[0]]['Xn'].shape[1]; print(len(tr), 'train', len(va), 'val', len(te), 'test clips; channels', CI, flush=True)
    def crop(k):
        X, y = C[k]['Xn'], C[k]['y']; n = len(y); s = math.exp(np.random.uniform(math.log(0.8), math.log(1.25))); L = int(CROP * s)
        a = np.random.randint(0, max(1, n - L)); idx = np.clip(np.round(np.linspace(a, a + L - 1, CROP)).astype(int), 0, n - 1)   # time stretch
        x = X[idx].copy(); t = y[idx].copy()
        if np.random.rand() < 0.5:                                # synthetic speed change: the water law (flow x f) and the target (+ log f)
            b0 = np.random.randint(0, CROP - 60); b1 = min(CROP, b0 + np.random.randint(60, 300)); lf = np.random.uniform(-0.5, 0.5)
            q = np.sinh(x[b0:b1, :NQ] * sd[:NQ] + mu[:NQ]) * math.exp(lf); x[b0:b1, :NQ] = (np.arcsinh(q) - mu[:NQ]) / sd[:NQ]; t[b0:b1] += lf
        x += np.random.normal(0, 0.05, x.shape).astype(np.float32)
        if np.random.rand() < 0.5: x[:, :NQ][:, np.random.rand(NQ) < 0.1] = 0.0   # profile-band dropout
        return x, t
    def batch(keys):
        xs, ys = zip(*[crop(keys[np.random.randint(len(keys))]) for _ in range(BATCH)])
        return torch.tensor(np.stack(xs)).permute(0, 2, 1).to(DEV), torch.tensor(np.stack(ys)).to(DEV)
    def loss_fn(p, y):
        m = torch.isfinite(y); y0 = torch.where(m, y, torch.zeros_like(y)); w = m.float(); n = w.sum(1, keepdim=True).clamp(min=1)
        pc = p - (p * w).sum(1, keepdim=True) / n; yc = y0 - (y0 * w).sum(1, keepdim=True) / n
        return (torch.nn.functional.smooth_l1_loss(pc, yc, reduction='none', beta=0.1) * w).sum() / w.sum().clamp(min=1)
    @torch.no_grad()
    def predict(model, k):
        model.eval(); x = torch.tensor(C[k]['Xn']).T[None].to(DEV)
        with torch.autocast('cuda', enabled=AMP): p = model(x)[0].float().cpu().numpy()
        model.train(); return p
    folds = [tr[i::5] for i in range(5)] if not REFIT else [[] for _ in range(5)]; PRED = {k: {} for k in C}; t0 = time.time(); CKPT = list(range(CK, EPOCHS + 1, CK))
    for fi, hold in enumerate(folds):
        keys = [k for k in tr if k not in hold]; torch.manual_seed(SEED * 100 + fi) if REFIT else None; model = TCN(CI).to(DEV)
        if fi == 0: print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
        opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.05); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EPOCHS * STEPS, pct_start=0.1)
        scaler = torch.cuda.amp.GradScaler(enabled=AMP)
        for ep in range(1, EPOCHS + 1):
            tot = 0.0
            for _ in range(STEPS):
                X, Y = batch(keys)
                with torch.autocast('cuda', enabled=AMP): l = loss_fn(model(X).float(), Y)
                opt.zero_grad(); scaler.scale(l).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); tot += float(l)
            if ep in CKPT:
                for k in hold: PRED[k][ep] = predict(model, k)
                for k in va + te: PRED[k].setdefault(ep, []).append(predict(model, k))
                print(f'fold {fi} epoch {ep} train loss {tot / STEPS:.4f} ({time.time() - t0:.0f} s)', flush=True)
    out = {f'{k}|{ep}': (np.mean(PRED[k][ep], 0) if isinstance(PRED[k][ep], list) else PRED[k][ep]) for k in C for ep in CKPT if ep in PRED[k]}   # refit: test only
    np.savez_compressed(f'{OUT}/big_pred_{TAG}.npz', **out); json.dump(dict(mu=mu.tolist(), sd=sd.tolist(), ckpt=CKPT, folds=folds, seed=SEED, refit=REFIT), open(f'{OUT}/big_meta_{TAG}.json', 'w'))
    def shape_err(keys, ep):
        e = []
        for k in keys:
            p = out[f'{k}|{ep}']; y = C[k]['y']; ok = np.isfinite(y)
            if ok.sum() < 200: continue
            W = [(np.mean(p[a:a + 160][ok[a:a + 160]]), np.mean(y[a:a + 160][ok[a:a + 160]])) for a in range(0, len(y) - 160, 160) if ok[a:a + 160].sum() > 80]
            if len(W) < 2: continue
            d = np.array([a - b for a, b in W]); e += list(np.abs(d - np.median(d)))
        return round(float(np.mean(e)), 4), len(e)
    for ep in (CKPT if not REFIT else []): print(f'epoch {ep}: window shape |err| train OOF {shape_err(tr, ep)}, val {shape_err(va, ep)}', flush=True)
    print('done', time.time() - t0, 's', flush=True)
