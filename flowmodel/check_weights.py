"""The published temporal-CNN weights (seed 0, epoch 60, 5 folds, variant allcam_nocam) on the CNN features of the test clips,
compared with the saved predictions of the same training run (big_pred_allcam_nocam.npz, key <clip>|60 = the mean of the 5 fold
models). Those were computed on a GPU with mixed precision; this runs in fp32 on the CPU, so the two agree to fp16 rounding.
usage: python flowmodel/check_weights.py [ARTIFACTS_DIR]"""
import sys, json, glob, os, numpy as np, torch, torch.nn as nn
A = sys.argv[1] if len(sys.argv) > 1 else 'artifacts'


class Block(nn.Module):   # as in train_seeds.py
    def __init__(s, h, k, d, p):
        super().__init__(); s.n = nn.GroupNorm(1, h); s.c1 = nn.Conv1d(h, h, k, padding=d * (k - 1) // 2, dilation=d); s.a = nn.GELU(); s.dr = nn.Dropout(p); s.c2 = nn.Conv1d(h, h, 1)
    def forward(s, x): return x + s.c2(s.dr(s.a(s.c1(s.n(x)))))
class TCN(nn.Module):
    def __init__(s, ci, h=320, k=7, dil=(1, 2, 4, 8, 16, 32, 1, 4), p=0.2):
        super().__init__(); s.inp = nn.Conv1d(ci, h, 1); s.blocks = nn.Sequential(*[Block(h, k, d, p) for d in dil]); s.out = nn.Sequential(nn.GroupNorm(1, h), nn.Conv1d(h, 1, 1))
    def forward(s, x): return s.out(s.blocks(s.inp(x)))[:, 0]


meta = json.load(open(f'{A}/flowmodel_predictions/big_meta_allcam_nocam.json')); mu, sd = np.float32(meta['mu']), np.float32(meta['sd'])
saved = np.load(f'{A}/flowmodel_predictions/big_pred_allcam_nocam.npz')
models = []
for f in range(5):
    m = TCN(len(mu)); m.load_state_dict(torch.load(f'{A}/flowmodel_weights_seed0/big_allcam_nocam_fold{f}_ep60.pt', map_location='cpu')); models.append(m.eval())
worst = []
for p in sorted(glob.glob(f'{A}/flowmodel_features/allcam/clip_*.npz')):
    z = np.load(p); k = os.path.basename(p)[:-4]
    if str(z['split']) != 'test': continue
    x = torch.tensor(np.nan_to_num((z['X'][:, :-6].astype(np.float32) - mu) / sd).astype(np.float32)).T[None]
    with torch.no_grad(): pr = np.mean([m(x)[0].numpy() for m in models], 0)
    ref = saved[f'{k}|60']; d = np.abs(pr - ref).max(); worst.append(d)
    print(f'{k}: max |weights - saved| {d:.2e} (saved prediction std {ref.std():.3f})')
print(f'{len(worst)} test clips: largest difference {max(worst):.2e}')
