"""Torch-free ByteTrack (SEA-6) — a faithful port of ifzhang/ByteTrack (MIT).

Ported from https://github.com/ifzhang/ByteTrack (MIT License) — the canonical
ByteTrack reference that ultralytics' and Qengineering's trackers both derive
from. The algorithm is reproduced in numpy + scipy only (no torch, no
ultralytics, no lap/cython_bbox):

  * KalmanFilterXYAH    — the DeepSORT 8-state (x,y,a,h,vx,vy,va,vh) constant-
                          velocity filter (kalman_filter.py upstream).
  * STrack              — per-track Kalman state; tentative (is_activated=False)
                          until a 2nd matching frame, so single-frame wave-FP
                          never get a returned id.
  * BYTETracker.update  — two-stage association (high-score IoU + fuse_score,
                          then low-score IoU against still-tracked tracks),
                          unconfirmed-track handling, and a lost-track buffer the
                          Kalman coasts so re-detections re-attach across gaps
                          (the part fast/sparse movers like boats need).

Adaptations vs upstream — none algorithmic:
  * scipy.optimize.linear_sum_assignment replaces lap.lapjv (the cost-limit is
    emulated exactly as ultralytics' scipy path does: solve, then drop pairs
    above the gate).
  * numpy IoU replaces cython_bbox.
  * I/O is the project's detection-dict shape in ({class, class_id, confidence,
    bbox=(x1,y1,x2,y2)}) and out ({track_id, class, confidence, bbox}); each
    track carries its class label + score from the latest matched detection.
  * Thresholds follow camera/bytetrack.yaml and the behaviour we tuned through
    the ultralytics path at person@0.25: '>=' high-split, new_track_thresh used
    directly, max_time_lost = track_buffer.
  * The track-id counter is per-tracker-instance (not upstream's class global),
    so a fresh ByteTracker() starts ids at 1.
"""
import time
import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment

try:
    import cv2  # GMC / camera-motion compensation; the camera stack always ships it
except Exception:  # pragma: no cover
    cv2 = None


class KalmanFilterXYAH:
    """8-state (x,y,a,h,vx,vy,va,vh) constant-velocity Kalman filter (DeepSORT)."""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros_like(measurement)]
        std = [2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],
               1e-2,
               2 * self._std_weight_position * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               1e-5,
               10 * self._std_weight_velocity * measurement[3]]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        std_pos = [self._std_weight_position * mean[3],
                   self._std_weight_position * mean[3],
                   1e-2,
                   self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3],
                   self._std_weight_velocity * mean[3],
                   1e-5,
                   self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [self._std_weight_position * mean[3],
               self._std_weight_position * mean[3],
               1e-1,
               self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                             (covariance @ self._update_mat.T).T,
                                             check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


_KF = KalmanFilterXYAH()   # stateless (matrices only) — safe to share across tracks


def _ious(atlbrs, btlbrs):
    """Vectorized IoU between two lists of (x1,y1,x2,y2) boxes -> [len(a), len(b)]."""
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    a = np.ascontiguousarray(atlbrs, dtype=float)
    b = np.ascontiguousarray(btlbrs, dtype=float)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0.0, None), axis=2)
    return inter / (area_a + area_b - inter + 1e-9)


def _iou_distance(atracks, btracks):
    return 1.0 - _ious([t.tlbr for t in atracks], [t.tlbr for t in btracks])


def _fuse_score(cost_matrix, detections):
    """Weight IoU similarity by detection confidence (ByteTrack fuse_score)."""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = np.array([d.score for d in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    return 1.0 - (iou_sim * det_scores)


def _linear_assignment(cost_matrix, thresh):
    """scipy min-cost assignment with a cost-limit gate (lap.lapjv emulation)."""
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])))
    matches, matched_r, matched_c = [], set(), set()
    rows, cols = linear_sum_assignment(cost_matrix)
    for r, c in zip(rows, cols):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c]); matched_r.add(r); matched_c.add(c)
    u_a = [r for r in range(cost_matrix.shape[0]) if r not in matched_r]
    u_b = [c for c in range(cost_matrix.shape[1]) if c not in matched_c]
    matches = np.asarray(matches) if matches else np.empty((0, 2), dtype=int)
    return matches, u_a, u_b


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """A single tracklet — Kalman state + the class/score of its latest detection."""

    def __init__(self, tlwh, score, cls, class_id):
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.mean = self.covariance = None
        self.is_activated = False
        self.score = score
        self.cls = cls
        self.class_id = class_id
        # Class is decided by an accumulated-confidence VOTE over the track's life
        # (not the single best frame): _cls_score[c] sums conf per class seen.
        self._cls_score = {cls: float(score)}
        self._cls_id = {cls: class_id}
        self.tracklet_len = 0
        self.state = TrackState.New
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0          # zero the height-velocity for non-tracked
        self.mean, self.covariance = _KF.predict(mean_state, self.covariance)

    @staticmethod
    def multi_gmc(stracks, H):
        """Warp the Kalman state's POSITION (x,y) and velocity (vx,vy) by the camera-
        motion affine H; leave aspect+height (a,h) untouched. Our filter is XYAH, NOT
        BoT-SORT's XYWH — applying H to the (a,h) pair would mangle the box (aspect ~1
        mixed with height ~80px by even a small rotation -> total ID fragmentation).
        Adapts BoT-SORT's STrack.multi_gmc to KalmanFilterXYAH."""
        if not len(stracks):
            return
        R = H[:2, :2]
        t = H[:2, 2]
        M = np.eye(8)
        M[0:2, 0:2] = R     # position (x, y)
        M[4:6, 4:6] = R     # velocity (vx, vy)
        for st in stracks:
            if st.mean is None:
                continue
            mean = M.dot(st.mean)
            mean[0:2] += t
            st.mean = mean
            if st.covariance is not None:
                st.covariance = M.dot(st.covariance).dot(M.T)

    def activate(self, frame_id, track_id):
        self.track_id = track_id
        self.mean, self.covariance = _KF.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1   # first-frame tracks are trusted at once
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id):
        self.mean, self.covariance = _KF.update(self.mean, self.covariance,
                                                self.tlwh_to_xyah(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self._adopt(new_track)

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = _KF.update(self.mean, self.covariance,
                                                self.tlwh_to_xyah(new_track.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self._adopt(new_track)

    def _adopt(self, new_track):
        self.score = new_track.score
        # Accumulate confidence per class, then label the track with the class that has
        # the most TOTAL evidence (frequency x confidence). Switch only when a challenger
        # clearly overtakes the current class (>10% hysteresis), so a single noisy or
        # high-conf wrong-class frame can't flip a stable track. A distant boat that
        # reads ship@0.35 a few frames but motorboat@0.5-0.9 many frames settles on
        # motorboat; association stays class-agnostic.
        c = new_track.cls
        self._cls_score[c] = self._cls_score.get(c, 0.0) + new_track.score
        self._cls_id[c] = new_track.class_id
        best = max(self._cls_score, key=self._cls_score.get)
        if best != self.cls and self._cls_score[best] > self._cls_score[self.cls] * 1.10:
            self.cls = best
            self.class_id = self._cls_id[best]

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @property
    def end_frame(self):
        return self.frame_id

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=float).copy()
        ret[2:] = np.maximum(ret[2:], 1.0)   # degenerate (0-area) box would NaN-poison the Kalman state
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr, dtype=float).copy()
        ret[2:] -= ret[:2]
        return ret


def _joint(tlista, tlistb):
    exists, res = {}, []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        if not exists.get(t.track_id, 0):
            exists[t.track_id] = 1
            res.append(t)
    return res


def _sub(tlista, tlistb):
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        stracks.pop(t.track_id, None)
    return list(stracks.values())


def _remove_duplicate(stracksa, stracksb):
    pdist = _iou_distance(stracksa, stracksb)
    dupa, dupb = [], []
    for p, q in zip(*np.where(pdist < 0.15)):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        (dupb if timep > timeq else dupa).append(q if timep > timeq else p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb


class GMC:
    """Global Motion Compensation — BoT-SORT's sparseOptFlow method (torch-free, cv2
    only): goodFeaturesToTrack + pyramidal LK + RANSAC give the inter-frame affine
    (camera motion), used to warp track predictions before association so a rocking
    (masthead) camera doesn't break IoU matching. Ported from
    github.com/niraharon/bot-sort (tracker/gmc.py). Active only when a frame is given."""

    # Skad brac cechy do dopasowania. Domyslnie z CALEGO kadru -- i to jest blad, bo
    # wiekszosc kadru to WODA, ktora plynie wzgledem lodzi, wiec afiniczna sledzi morze,
    # a nie kamere. Zmierzone na 26 ciagach archiwum (958 klatek), blad przewidzenia
    # polozenia punktu ODLEGLEJ SCENY w nastepnej klatce, na punktach trzymanych z boku:
    #     cechy z wody          14.52 px
    #     cechy z calego kadru   7.89 px   <- zachowanie sprzed tej zmiany
    #     cechy z nieba          3.13 px   <- 2.5x lepiej
    # Dryf zlozony przy dopasowaniu z nieba jest prawie plaski: 3.13 / 4.04 / 4.57 / 4.75 /
    # 6.77 px po 1/2/3/5/8 krokach (~8.4 s), wiec to nie jest zysk tylko na jednej klatce.
    SKY_MARGIN_FRAC = 0.04    # ile NAD horyzontem zaczyna sie maska (ulamek wysokosci kadru)
    MIN_SKY_FRAC = 0.06       # ponizej tylu % kadru nieba maska nie ma sensu
    MIN_SKY_KP_FRAC = 0.1     # ponizej tylu % z maxCorners wracamy do calego kadru

    def __init__(self, downscale=2):
        self.downscale = max(1, int(downscale))
        self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1,
                                   blockSize=3, useHarrisDetector=False, k=0.04)
        self.prev_gray = None
        self.prev_kp = None

    def _sky_mask(self, gray, horizon_y):
        """Maska nad horyzontem, albo None gdy nieba za malo/nie wiadomo gdzie jest.

        `horizon_y` jest w pikselach PELNEJ klatki -- skalujemy je tak samo jak obraz.
        Wolajacy podaje NAJWYZSZY horyzont z kilku kolumn, bo pod przechylem horyzont
        jest ukosny i jeden wiersz nie opisuje calego kadru."""
        if horizon_y is None:
            return None
        cut = int(horizon_y) // self.downscale - int(self.SKY_MARGIN_FRAC * gray.shape[0])
        if cut < self.MIN_SKY_FRAC * gray.shape[0]:
            return None
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[:cut, :] = 255
        return mask

    def apply(self, frame, horizon_y=None):
        H = np.eye(2, 3, dtype=float)
        if cv2 is None or frame is None:
            return H
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if self.downscale > 1:
            gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))
        mask = self._sky_mask(gray, horizon_y)
        kp = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        # Puste niebo (bezchmurny dzien, horyzont poza kadrem pod przechylem) nie daje cech.
        # Wtedy WRACAMY do calego kadru -- gorsza rejestracja jest lepsza niz zadna, a to
        # jest dokladnie zachowanie sprzed tej zmiany, wiec fallback nie moze nic zepsuc.
        prog = self.MIN_SKY_KP_FRAC * self.feature_params["maxCorners"]
        if mask is not None and (kp is None or len(kp) < prog):
            kp = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        if kp is None:
            kp = np.empty((0, 1, 2), dtype=np.float32)
        if self.prev_gray is None or self.prev_kp is None or len(self.prev_kp) < 5:
            self.prev_gray, self.prev_kp = gray, kp
            return H
        matched, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_kp, None)
        if matched is not None and status is not None:
            good = status.ravel() == 1
            prev_pts, curr_pts = self.prev_kp[good], matched[good]
            if len(prev_pts) > 4:
                M, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)
                if M is not None:
                    H = M.astype(float)
                    if self.downscale > 1:
                        H[0, 2] *= self.downscale
                        H[1, 2] *= self.downscale
        self.prev_gray, self.prev_kp = gray, kp
        return H


class ByteTracker:
    """Faithful ByteTrack — update() takes/returns the project's detection dicts."""

    def __init__(self, high_thresh=0.25, low_thresh=0.05, new_thresh=0.25,
                 match_thresh=0.8, track_buffer=30, fuse_score=True,
                 cmc=True, cmc_downscale=2):
        self.high = high_thresh
        self.low = low_thresh
        self.new = new_thresh
        self.match_thresh = match_thresh
        self.fuse = fuse_score
        self.max_time_lost = track_buffer
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.fazy = {}       # rozbicie update() na etapy; zbiera je wachta
        self._count = 0
        # Camera-motion compensation (BoT-SORT GMC). Built here but only does work when
        # a frame is passed to update() — so frame-less callers (tests) are unaffected.
        self.gmc = GMC(downscale=cmc_downscale) if (cmc and cv2 is not None) else None

    def _next_id(self):
        self._count += 1
        return self._count

    @staticmethod
    def _to_strack(d):
        return STrack(STrack.tlbr_to_tlwh(d["bbox"]), d["confidence"], d["class"], d.get("class_id"))

    def _etap(self, nazwa, t0):
        """Zamyka etap `nazwa` i zwraca znacznik dla nastepnego -- jak w watchkeeperze.

        Wlasny licznik, a nie ten z wachty: `byte_track` nie importuje `watchkeeper`
        (bylby cykl), a szesc linii jest tansze niz wspolny modul dla jednego uzycia.
        Wachta zbiera `self.fazy` i wystawia je w telemetrii z przedrostkiem `bt_`.
        """
        teraz = time.perf_counter()
        self.fazy[nazwa] = self.fazy.get(nazwa, 0.0) + (teraz - t0)
        return teraz

    def update(self, dets, frame=None, horizon_y=None):
        _e = time.perf_counter()      # lancuch etapow -- patrz `_etap`
        self.frame_id += 1
        activated, refind, lost, removed = [], [], [], []

        scores = np.array([d["confidence"] for d in dets], dtype=float)
        remain = scores >= self.high
        inds_second = (scores > self.low) & (scores < self.high)
        detections = [self._to_strack(dets[i]) for i in np.where(remain)[0]]
        detections_second = [self._to_strack(dets[i]) for i in np.where(inds_second)[0]]

        unconfirmed, tracked = [], []
        for t in self.tracked_stracks:
            (tracked if t.is_activated else unconfirmed).append(t)

        # Step 1: first association on high-score detections (IoU, score-fused).
        strack_pool = _joint(tracked, self.lost_stracks)
        _e = self._etap('przygotowanie', _e)
        for t in strack_pool:
            t.predict()
        # Camera-motion compensation: warp predictions by the inter-frame affine so
        # they stay aligned to detections despite camera motion (BoT-SORT GMC).
        _e = self._etap('predykcja', _e)
        if frame is not None and self.gmc is not None:
            warp = self.gmc.apply(frame, horizon_y=horizon_y)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)
        _e = self._etap('gmc', _e)
        dists = _iou_distance(strack_pool, detections)
        if self.fuse:
            dists = _fuse_score(dists, detections)
        matches, u_track, u_det = _linear_assignment(dists, self.match_thresh)
        for it, idet in matches:
            track, det = strack_pool[it], detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id); activated.append(track)
            else:
                track.re_activate(det, self.frame_id); refind.append(track)

        # Step 2: second association on low-score detections vs still-tracked tracks.
        _e = self._etap('skojarzenie1', _e)
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        matches, u_track2, _ = _linear_assignment(_iou_distance(r_tracked, detections_second), 0.5)
        for it, idet in matches:
            track, det = r_tracked[it], detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id); activated.append(track)
            else:
                track.re_activate(det, self.frame_id); refind.append(track)
        for it in u_track2:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost(); lost.append(track)

        # Unconfirmed tracks (a single beginning frame): match a leftover high det or drop.
        _e = self._etap('skojarzenie2', _e)
        detections = [detections[i] for i in u_det]
        dists = _iou_distance(unconfirmed, detections)
        if self.fuse:
            dists = _fuse_score(dists, detections)
        matches, u_unconfirmed, u_det = _linear_assignment(dists, 0.7)
        for it, idet in matches:
            unconfirmed[it].update(detections[idet], self.frame_id); activated.append(unconfirmed[it])
        for it in u_unconfirmed:
            unconfirmed[it].mark_removed(); removed.append(unconfirmed[it])

        # Step 4: init new tracks from leftover high detections above new_thresh.
        _e = self._etap('niepotwierdzone', _e)
        for inew in u_det:
            track = detections[inew]
            if track.score < self.new:
                continue
            track.activate(self.frame_id, self._next_id())
            activated.append(track)

        # Step 5: expire lost tracks past the buffer.
        _e = self._etap('nowe_tory', _e)
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed(); removed.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = _joint(self.tracked_stracks, activated)
        self.tracked_stracks = _joint(self.tracked_stracks, refind)
        self.lost_stracks = _sub(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = _sub(self.lost_stracks, removed)
        self.removed_stracks = removed   # reset each frame — no unbounded growth
        self.tracked_stracks, self.lost_stracks = _remove_duplicate(self.tracked_stracks, self.lost_stracks)

        _e = self._etap('porzadki', _e)
        out = []
        for t in self.tracked_stracks:
            if not t.is_activated:
                continue
            x1, y1, x2, y2 = t.tlbr
            out.append({"track_id": t.track_id, "class": t.cls, "confidence": t.score,
                        "bbox": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))})
        self._etap('wyjscie', _e)
        return out
