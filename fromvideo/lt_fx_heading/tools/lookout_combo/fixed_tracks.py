"""Tracks on things FIXED TO THE BOAT (antenna, rail, people on the bow, lens drops), GPS-free.
A far scene point moves in the image whenever the camera rotates -- turning OR wave pitch/roll; a boat-fixed point stays
put. So judge only INFORMATIVE windows (DT steps): the scene moves, i.e. the 75th percentile of all tracks' displacement
over the window > MOVE px (a percentile, not the median, so a majority of fixed points cannot hide the scene motion).
In such a window a track is 'still' if it moved < max(STILL px, FRAC * that percentile). Flag a track as boat-fixed when
it was still in >= SHARE of its informative windows and had >= MIN_W of them; tracks never in an informative window are
kept (when nothing moves they cannot bias the course either).
Library: fixed_track_ids(rows, rule) -> set of track ids; rule 'still' (above), 'follow' (follow_track_ids: also drops,
people) or 'slow' ('follow' limited to tracks that also move little: drops, boat, overlays; NOT near scene objects).
[--follow / --slow select the rule in the CLI.] CLI: per-clip share for train/test flir-VIS, and --draw clip frame
-> PNG with kept (green) and flagged (red) tracks on that frame."""
import sys, glob, os, numpy as np
DT, MOVE, STILL, FRAC, SHARE, MIN_W = 8, 3.0, 0.7, 0.15, 0.8, 2
G_MIN, H_THR, R_SLOW = 0.5, 3.0, 0.5
def follow_track_ids(rows, slow=None):
    """Rule 'follow' (drops sliding on the lens, people walking on deck, plus everything 'still' catches): the camera
    rocks on the waves all the time and every SCENE point -- near, far, even clouds -- moves with that motion; things on
    the lens or on the boat do not. Per DT-step window: the scene's image motion = a RANSAC homography (H_THR px) fitted
    to all tracks; informative if its median predicted displacement > MOVE px. Per track over its informative windows:
    g = sum(d . m) / sum(m . m) (its displacement d against the scene's predicted displacement m at its position);
    scene ~1, lens/boat ~0. Flag g < G_MIN with >= MIN_W informative windows.
    slow=R: flag only if ALSO sum|d| / sum|m| < R. Rule 'follow' alone flagged NEAR scene objects too (clip_000 under a
    bridge: the pillar and its sign, the bridge's underside): with the boat's forward motion they move by parallax, often
    MORE than the far scene but in another direction, so g < 0.5 -- yet they carry the translation E16 needs. Tracks
    that move but not like the scene (near objects, people walking) are left to RANSAC / the BA second pass."""
    import cv2
    fr = rows[:, 0].astype(int); tid = rows[:, 1].astype(int); xy = rows[:, 2:4].astype(float)
    pos = {(t, f): p for t, f, p in zip(tid, fr, xy)}
    by_frame = {}
    for t, f in zip(tid, fr): by_frame.setdefault(f, []).append(t)
    num, den, nw, ad, am = {}, {}, {}, {}, {}
    for f0 in range(0, fr.max() - DT + 1, DT):
        ts = [t for t in by_frame.get(f0, []) if (t, f0 + DT) in pos]
        if len(ts) < 10: continue
        a = np.array([pos[(t, f0)] for t in ts], np.float32); b = np.array([pos[(t, f0 + DT)] for t in ts], np.float32)
        cv2.setRNGSeed(0); H, _ = cv2.findHomography(a, b, cv2.RANSAC, H_THR)
        if H is None: continue
        pr = cv2.perspectiveTransform(a.reshape(-1, 1, 2), H).reshape(-1, 2); m = pr - a; d = b - a
        if np.median(np.hypot(m[:, 0], m[:, 1])) <= MOVE: continue
        for t, di, mi in zip(ts, d, m):
            num[t] = num.get(t, 0.0) + float(di @ mi); den[t] = den.get(t, 0.0) + float(mi @ mi); nw[t] = nw.get(t, 0) + 1
            ad[t] = ad.get(t, 0.0) + float(np.hypot(*di)); am[t] = am.get(t, 0.0) + float(np.hypot(*mi))
    return {t for t in nw if nw[t] >= MIN_W and num[t] / den[t] < G_MIN and (slow is None or ad[t] / am[t] < slow)}
def fixed_track_ids(rows, rule='still'):
    if rule == 'follow':
        return follow_track_ids(rows)
    if rule == 'slow':
        return follow_track_ids(rows, slow=R_SLOW)
    fr = rows[:, 0].astype(int); tid = rows[:, 1].astype(int); xy = rows[:, 2:4].astype(float)
    pos = {}                                                  # (track, frame) -> xy
    for t, f, p in zip(tid, fr, xy): pos[(t, f)] = p
    by_frame = {}
    for t, f in zip(tid, fr): by_frame.setdefault(f, []).append(t)
    info, still = {}, {}
    for f0 in range(0, fr.max() - DT + 1, DT):
        ts = [t for t in by_frame.get(f0, []) if (t, f0 + DT) in pos]
        if len(ts) < 10: continue
        d = np.array([np.hypot(*(pos[(t, f0 + DT)] - pos[(t, f0)])) for t in ts])
        p75 = np.percentile(d, 75)
        if p75 <= MOVE: continue
        lim = max(STILL, FRAC * p75)
        for t, di in zip(ts, d):
            info[t] = info.get(t, 0) + 1; still[t] = still.get(t, 0) + (di < lim)
    return {t for t in info if info[t] >= MIN_W and still[t] >= SHARE * info[t]}
if __name__ == '__main__':
    RULE = 'follow' if '--follow' in sys.argv else 'slow' if '--slow' in sys.argv else 'still'
    if '--draw' in sys.argv:
        import cv2, json
        k, f = sys.argv[sys.argv.index('--draw') + 1], int(sys.argv[sys.argv.index('--draw') + 2])
        src = 'runs/lt_rot_test' if os.path.exists(f'runs/lt_rot_test/{k}.npz') else 'runs/lt_rot'
        root = 'data/test' if 'test' in src else 'data/release'
        sys.path.insert(0, root + '/starter')
        from preprocessing import ClipPreprocessor
        from public_dataset import load_manifest, load_timestamps
        c = {c['key']: c for c in load_manifest(root + '/release_manifest.json')['clips']}[k]
        pre = ClipPreprocessor(root + '/' + c['video'], c['camera_mode'], 8.0, half_res=True, calibration=c['calibration'],
                               frame_timestamps=load_timestamps(root, c))
        img = next(im for i, (_, im, _, _, _) in enumerate(pre.frames_with_time()) if i == f).copy()
        rows = np.load(f'{src}/{k}.npz')['rows']; fx = fixed_track_ids(rows, RULE)
        here = rows[rows[:, 0].astype(int) == f]
        for t, x, y in zip(here[:, 1].astype(int), here[:, 2], here[:, 3]):
            cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255) if t in fx else (0, 255, 0), 2 if t in fx else 1)
        out = f'fixed_{RULE}_{k}_{f}.png'; cv2.imwrite(out, img)
        print(out, f'{sum(t in fx for t in here[:, 1].astype(int))} flagged / {len(here)} tracks on this frame')
        sys.exit()
    for name, pat in (('train flir-VIS', 'runs/lt_rot/clip_*.npz'), ('test flir-VIS', 'runs/lt_rot_test/clip_*.npz')):
        print(name)
        for f in sorted(glob.glob(pat)):
            rows = np.load(f)['rows']; fx = fixed_track_ids(rows, RULE); tid = rows[:, 1].astype(int)
            obs = np.isin(tid, list(fx)).mean() if fx else 0.0
            best = np.bincount(rows[np.isin(tid, list(fx)), 0].astype(int)).argmax() if fx else -1
            print(f'  {os.path.basename(f)[:-4]}  flagged tracks {len(fx):5d} / {len(set(tid)):6d}  = {obs:5.1%} of observations;'
                  f' frame with most flagged: {best}')
