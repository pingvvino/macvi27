"""Maritime post-processing pipeline for YOLO detections.
Soft-NMS, horizon detection/masking, and tiled-detection merge.
"""
import math
import numpy as np
import cv2


# ── Soft-NMS ───────────────────────────────────────────────────────────────────

def soft_nms(detections: list[dict], sigma: float = 0.5, score_thresh: float = 0.15) -> list[dict]:
    """Gaussian Soft-NMS: instead of hard suppression, decay confidence of
    overlapping boxes. Preserves partially-overlapping objects that hard NMS kills.

    Args:
        detections: list of dicts with 'bbox' (x1,y1,x2,y2) and 'confidence'
        sigma: Gaussian decay parameter (lower = more aggressive)
        score_thresh: discard detections below this after decay

    Returns:
        Filtered detections with updated confidence scores
    """
    if not detections:
        return detections

    N = len(detections)
    boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    scores = np.array([d["confidence"] for d in detections], dtype=np.float32)
    indices = list(range(N))

    result_indices = []

    for i in range(N):
        max_idx = i + np.argmax(scores[i:])
        # Swap
        boxes[[i, max_idx]] = boxes[[max_idx, i]]
        scores[[i, max_idx]] = scores[[max_idx, i]]
        indices[i], indices[max_idx] = indices[max_idx], indices[i]

        result_indices.append(indices[i])

        ix1, iy1, ix2, iy2 = boxes[i]
        i_area = (ix2 - ix1) * (iy2 - iy1)

        for j in range(i + 1, N):
            jx1, jy1, jx2, jy2 = boxes[j]
            j_area = (jx2 - jx1) * (jy2 - jy1)

            xx1 = max(ix1, jx1)
            yy1 = max(iy1, jy1)
            xx2 = min(ix2, jx2)
            yy2 = min(iy2, jy2)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            union = i_area + j_area - inter
            iou = inter / union if union > 0 else 0

            # Gaussian decay
            scores[j] *= math.exp(-(iou * iou) / sigma)

    # Filter by score threshold and rebuild
    output = []
    for idx, orig_idx in enumerate(result_indices):
        if scores[idx] >= score_thresh:
            det = dict(detections[orig_idx])
            det["confidence"] = round(float(scores[idx]), 3)
            output.append(det)

    return output


# ── Horizon Masking ────────────────────────────────────────────────────────────

def detect_horizon(frame: np.ndarray, fallback_ratio: float = 0.55) -> int:
    """Detect horizon line in maritime scene using edge detection + Hough transform.
    Returns y-coordinate of horizon. Falls back to fixed ratio if detection fails.

    On a yacht, horizon is typically in the upper 40-60% of the frame depending
    on camera mount angle and sea state.
    """
    h, w = frame.shape[:2]
    fallback_y = int(h * fallback_ratio)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    # Look for dominant horizontal line in upper 70% of image
    roi = edges[:int(h * 0.7), :]
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=100,
                            minLineLength=w // 3, maxLineGap=w // 10)

    if lines is None:
        return fallback_y

    # Find the most horizontal line (slope near 0)
    best_y = fallback_y
    best_score = -1

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < w * 0.2:
            continue
        slope = abs(y2 - y1) / max(abs(x2 - x1), 1)
        # Score: longer + more horizontal = better
        if slope < 0.1:
            score = length * (1.0 - slope)
            if score > best_score:
                best_score = score
                best_y = (y1 + y2) // 2

    return best_y


def horizon_curve(img, step=32, min_edge=4.0):
    """The sea/sky boundary row for each sampled column, or None if the frame has no clean one.

    A 160 degree fisheye bends the horizon into an ARC: measured off Wladyslawowo, row 310 at the
    left edge, 138 at centre, 374 at the right -- 254 px peak to peak. A straight line through
    that leaves 65 px rms, which is why the line-fitting estimator scores ~0.13 on frames whose
    horizon is plainly visible and correctly located.

    Rather than model the arc, this just reports what is measured, column by column; placing a
    marker only needs the row at the marker's own column. NOTE: the arc IS modelable -- see
    horizon_arc_candidates() above, which fits it to 1.3-3 px. The 454 px rms recorded here was
    an ANALYTIC derivation, not an empirical parabola fit, and should not be read as evidence
    that the geometry is out of reach.

    Returns (xs, ys) as parallel lists, or None.
    """
    import numpy as np
    h, w = img.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    band = g[0:int(h * 0.55), :]
    xs, ys = [], []
    for x in range(step, w - step, step):
        d = np.abs(np.diff(band[:, x]))
        if d.size < 10:
            continue
        y = int(np.argmax(d))
        if d[y] < min_edge:
            continue
        xs.append(x)
        ys.append(float(y))
    if len(xs) < 8:
        return None
    ys_arr = np.array(ys)
    med = np.median(ys_arr)
    keep = np.abs(ys_arr - med) < h * 0.2            # drop columns that latched onto a cloud
    xs = [x for x, k in zip(xs, keep) if k]
    ys = [y for y, k in zip(ys, keep) if k]
    return (xs, ys) if len(xs) >= 8 else None


# --- Horizon as an ARC -------------------------------------------------------
# A 160-degree fisheye bends the horizon into a parabola, and it is modelable --
# contrary to the note in horizon_curve() below, which reports a failed ANALYTIC
# derivation, not a failed empirical fit. Measured 2026-08-23 on 272 hand-drawn
# horizon points across six annotation jobs plus full-width shoreline traces:
#
#   y(x) = A + B*(x - W/2) + C*(x - W/2)^2      residual 1.3-3 px
#   a straight line through the same points     residual 20-52 px
#
# C is NOT a constant: it shrinks as the horizon sits lower in the frame, because
# a fisheye bends a line less near the optical axis than far from it. Measured on
# 1426 accepted fits: A=0-60 -> 1.65e-4, 60-120 -> 1.56e-4, 120-180 -> 1.48e-4,
# 180-260 -> 1.30e-4. Predicting C from A leaves 8.8e-6 scatter, versus 1.12e-5
# for a fixed C -- and a FREE C runs off to 3e-4 on a hazy horizon and wrecks the
# shape. So we never search for C; we derive it from A.
# The measurements above were taken on 2560px-wide frames; the live camera serves
# 800x448. Curvature is not scale-free: halve the image and C doubles, because y
# and x shrink together while C multiplies x squared. So convert both ways.
_ARC_CA, _ARC_CB = -2.271e-07, 1.803e-04
_ARC_REF_W = 2560.0
# Polszerokosc korytarza, w ktorym szukamy horyzontu wzdluz przewidzianego luku, przy
# kadrze wysokosci 1440. Przemiat na 46 kadrach recznej prawdy (mediana bledu):
#     90 px -> 9,3   100 -> 7,8   110 -> 4,8   120 -> 4,7   130 -> 7,5   160 -> 13,7
# Krzywa jest gladka w obie strony, wiec to nie jest szum probki. Za waski gubi horyzont
# przy krawedziach kadru (blad samego modelu siega tam 250 px), za szeroki wpuszcza fale.
_KORYTARZ_POLOWA = 110.0


def arc_curvature(row: float, frame_w: float = _ARC_REF_W, heel_deg=None) -> float:
    """Horizon curvature C for a horizon at image row `row` in a frame `frame_w` wide.

    `heel_deg` (from the boat's AHRS) corrects for the boat lying over. The same arc,
    written as y(x) in a ROTATED frame, has a larger d^2 coefficient -- by 1/cos^3(theta).
    Row alone cannot express that, and the clamp below then truncates it: measured on 16
    hand-drawn horizons at 32-43 deg of image tilt, the true C ran 2.23e-4 to 4.52e-4
    (median 2.76e-4) while this function returned 1.44-1.75e-4 -- ALL SIXTEEN above the
    ceiling, median ratio 1.65. On the 46 hand-drawn frames together (0.4-28.9 deg) the
    sec^3 term halves the curvature error, median |dC| 3.40e-5 -> 1.74e-5, and end to end
    in pixels it is better everywhere and worse nowhere (all 46: median 11.5 -> 10.8 px,
    mean 140 -> 114; the 16 heeled ones: 60.4 -> 50.8 px, mean 328 -> 254).

    Take theta from the HEEL, never from the fit's own B: at high heel with glare the fit
    misreads the rotation by 20 deg (sensor 21.0 deg, fit -41.0 deg on 16-23-13), and
    sec^3 of that overshoots C to 3.15e-4 against a truth of 2.76e-4. `heel_to_slope`
    predicts the drawn angle to 1.05 deg median over the same 46 frames.

    A free-C fit is still not allowed (see above); the correction is a known multiplier.
    """
    k = _ARC_REF_W / max(1.0, float(frame_w))          # frame -> reference scale
    c_ref = _ARC_CA * float(row) * k + _ARC_CB
    c_ref = max(1.15e-4, min(1.75e-4, c_ref))
    b = heel_to_slope(heel_deg)
    if b is not None:                                  # clamp first, THEN heel
        c_ref = c_ref / max(math.cos(math.atan(b)) ** 3, 0.15)
    return c_ref * k


def _okno_kolumny(x, w, korytarz, y0, y1, h):
    """(y0, y1) dla JEDNEJ kolumny: albo wspolny pas, albo korytarz wzdluz luku."""
    if korytarz is None:
        return y0, y1
    (A, B, C), polowa = korytarz
    d = x - w / 2.0
    yc = A + B * d + C * d * d
    return max(0, int(yc - polowa)), min(h, int(yc + polowa))


def _okna_wektorowo(w, step, korytarz, y0, y1, h):
    """(xs, a, b) dla WSZYSTKICH probkowanych kolumn naraz -- odpowiednik `_okno_kolumny`.

    Ten sam wzor i to samo obcinanie do `int()` (numpy tez ucina do zera), zeby szybka
    sciezka i petla dawaly identyczne okna. Rozjazd tutaj bylby niewidoczny: luk nadal by
    sie dopasowal, tylko na innych pikselach.
    """
    xs = np.arange(step // 2, w, step)
    if korytarz is None:
        return xs, np.full(xs.shape, int(y0), np.int64), np.full(xs.shape, int(y1), np.int64)
    (A, B, C), polowa = korytarz
    d = xs - w / 2.0
    yc = A + B * d + C * d * d
    return (xs,
            np.maximum(0, (yc - polowa).astype(np.int64)),
            np.minimum(h, (yc + polowa).astype(np.int64)))


def _srednie_poziome(g, r0, r1, xs, step, w):
    """(r1-r0, n) float64 -- poziome srednie okien `step+1` px wokol KAZDEJ kolumny `xs`.

    Widokiem o przeskokach (`as_strided`), nie cumsum po szerokosci: cumsum float64 na pasie
    472x2560 to ZMIERZONE ~6 ms, czyli WIECEJ, niz kosztuje petla, ktora ma zastapic. Widok
    nie kopiuje nic, alokuje sie dopiero wynik usrednienia. `xs` musi byc regularne
    (co `step`, od `step//2`) -- i jest, bo tak je tworzy `_okna_wektorowo`.
    """
    szer = step // 2 * 2 + 1
    pas = np.ascontiguousarray(g[r0:r1])
    s0, s1 = pas.strides
    lew = xs - step // 2
    n_pe = int(np.searchsorted(lew, w - szer, side="right"))
    czesci = []
    if n_pe:
        widok = np.lib.stride_tricks.as_strided(
            pas, shape=(pas.shape[0], n_pe, szer), strides=(s0, step * s1, s1))
        czesci.append(widok.mean(axis=2, dtype=np.float64))
    for x in xs[n_pe:]:                      # ogon przy prawej krawedzi: okno przyciete
        czesci.append(pas[:, max(0, x - step // 2):x + step // 2 + 1]
                      .mean(axis=1, dtype=np.float64)[:, None])
    return czesci[0] if len(czesci) == 1 else np.concatenate(czesci, axis=1)


def _macierz_kolumn(sr, r0, a, H, kol):
    """(H, len(kol)) -- z (wiersze pasa, n) wycina kazdej kolumnie jej wlasne okno."""
    idx = (a - r0)[None, :] + np.arange(H)[:, None]
    return sr[idx, kol[None, :]]


def _grupy_okien(a, b, minimalna):
    """[(maski, H), ...] -- kolumny pogrupowane po WYSOKOSCI okna, albo None.

    Wysokosci rozjezdzaja sie o 1 px, bo `polowa` korytarza nie jest calkowita, wiec
    `int(yc - polowa)` i `int(yc + polowa)` ucinaja rozne czesci ulamkowe. ZMIERZONE na
    realnym kadrze: 219..220, czyli DWIE grupy. Wymaganie rownosci wszedzie zabijalo
    szybka sciezke w dwoch z trzech wywolan -- stad grupy zamiast jednego progu.
    Przy wiekszym rozrzucie (przyciecia krawedzia kadru) wracamy do petli.
    """
    wys = b - a
    dobre = wys >= minimalna
    if dobre.sum() < 12:
        return None
    unik = np.unique(wys[dobre])
    if len(unik) > 3:
        return None
    return [(dobre & (wys == int(hh)), int(hh)) for hh in unik]


def _arc_trace_edge(g, y0, y1, step=8, half=12, korytarz=None):
    """Per column: row of the strongest bright-over-dark step, and its strength.

    `korytarz` = ((A, B, C), polowa) szuka w PASMIE WZDLUZ przewidzianego luku zamiast
    we wspolnym pasie poziomym. Pas poziomy przy przechylu musi objac cala rozpietosc
    luku -- ponad 1100 px przy 25 st. -- wiec wciaga do sladu pasmo fal, ktore nigdy
    nie beda pasowac i psuja zarowno dopasowanie, jak i rezyduum. Korytarz +-110 px
    zmierzony na 46 kadrach recznej prawdy: mediana bledu 9,3 -> 4,8 px, srednia
    68,0 -> 22,4; kadrow bez luku (2532 kadry archiwum) 6,2% -> 4,9%, a przy przechyle
    >= 15 st. 14,4% -> 8,3%. Na Pi 145,6 -> 21,7 ms przy 2560x1440, bo kolumna ma
    220 px zamiast calego pasa.
    """
    h, w = g.shape[:2]
    y0 = max(0, int(y0)); y1 = min(h, int(y1))
    if korytarz is None and y1 - y0 < 3 * half:
        return None
    # ── szybka sciezka: 320 obrotow petli Pythona -> kilka operacji na macierzy (H, n) ──
    # ZMIERZONE na Pi (2560x1440, pas 500 px): ten slad 16,8 ms, slad Otsu 34,3 ms, razem
    # 86% kosztu `horizon_arc_tiles` -- a rozmycie i konwersja w OpenCV tylko 17%. Skalowanie
    # obrazu atakowaloby te tanie 17% i placilo za to ziarnistoscia sladu; tu nie zmienia sie
    # ANI JEDEN piksel wejscia, tylko sposob przejscia po nich.
    xs_all, a_all, b_all = _okna_wektorowo(w, step, korytarz, y0, y1, h)
    grupy = _grupy_okien(a_all, b_all, 3 * half)
    if grupy is not None:
        r0 = int(a_all[b_all - a_all >= 3 * half].min())
        r1 = int(a_all[b_all - a_all >= 3 * half].max()) + max(H for _, H in grupy)
        sr = _srednie_poziome(g, r0, min(r1, h), xs_all, step, w)
        wx, wy, ws = [], [], []
        for maska, H in grupy:
            kol = np.nonzero(maska)[0]
            COL = _macierz_kolumn(sr, r0, a_all[maska], H, kol)
            cs = np.zeros((H + 1, COL.shape[1]), dtype=np.float64)
            np.cumsum(COL, axis=0, out=cs[1:])
            # np.convolve(col, k, "valid") z tym jadrem to dokladnie: srednia `half` wierszy
            # GORNYCH minus srednia `half` DOLNYCH. Liczymy to z cumsum zamiast splotem.
            sc = ((cs[half:H - half + 1] - cs[0:H - 2 * half + 1]) -
                  (cs[2 * half:H + 1] - cs[half:H - half + 1])) / half
            i = np.argmax(sc, axis=0)
            wx.append(xs_all[maska]); wy.append(a_all[maska] + i + half)
            ws.append(sc[i, np.arange(sc.shape[1])])
        x_, y_, s_ = np.concatenate(wx), np.concatenate(wy), np.concatenate(ws)
        kolejnosc = np.argsort(x_, kind="stable")     # petla oddaje kolumny rosnaco
        if len(x_) < 12:
            return None
        return (x_[kolejnosc].astype(np.float64), y_[kolejnosc].astype(np.float64), s_[kolejnosc])
    k = np.concatenate([np.full(half, 1.0 / half), np.full(half, -1.0 / half)])[::-1]
    xs, ys, ss = [], [], []
    for x in range(step // 2, w, step):
        a, b = _okno_kolumny(x, w, korytarz, y0, y1, h)
        if b - a < 3 * half:
            continue
        col = g[a:b, max(0, x - step // 2):x + step // 2 + 1].mean(axis=1)
        sc = np.convolve(col, k, mode="valid")
        if len(sc) == 0:
            continue
        i = int(np.argmax(sc))
        xs.append(float(x)); ys.append(float(a + i + half)); ss.append(float(sc[i]))
    if len(xs) < 12:
        return None
    return np.array(xs), np.array(ys), np.array(ss)


def _arc_trace_otsu(g, y0, y1, step=8, korytarz=None):
    """Per column: row that best SPLITS the column into sky/water, Otsu-style.

    Unlike the edge trace this looks at every pixel above and below rather than a
    24px neighbourhood, so it survives haze where the edge itself is washed out.
    Criterion: n1*n2/N^2 * (mean_above - mean_below)^2. After Miranda Neto et al.,
    "Robust Horizon Finding Algorithm for Real-Time Autonomous Navigation based on
    Monocular Vision", IEEE ITSC 2011.
    """
    h, w = g.shape[:2]
    y0 = max(0, int(y0)); y1 = min(h, int(y1))
    if korytarz is None and y1 - y0 < 20:
        return None
    # ── szybka sciezka, patrz `_arc_trace_edge` ──
    xs_all, a_all, b_all = _okna_wektorowo(w, step, korytarz, y0, y1, h)
    grupy = _grupy_okien(a_all, b_all, 20)
    if grupy is not None:
        r0 = int(a_all[b_all - a_all >= 20].min())
        r1 = int(a_all[b_all - a_all >= 20].max()) + max(H for _, H in grupy)
        sr = _srednie_poziome(g, r0, min(r1, h), xs_all, step, w)
        wx, wy, ws = [], [], []
        for maska, H in grupy:
            kol = np.nonzero(maska)[0]
            cs = np.cumsum(_macierz_kolumn(sr, r0, a_all[maska], H, kol), axis=0)
            n1 = np.arange(1, H, dtype=np.float64)[:, None]
            n2 = H - n1
            moc = (n1 * n2 / float(H * H)) * (cs[:-1] / n1 - (cs[-1] - cs[:-1]) / n2) ** 2
            i = np.argmax(moc, axis=0)
            wx.append(xs_all[maska]); wy.append(a_all[maska] + i + 1)
            ws.append(moc[i, np.arange(moc.shape[1])])
        x_, y_, s_ = np.concatenate(wx), np.concatenate(wy), np.concatenate(ws)
        if len(x_) < 12:
            return None
        kolejnosc = np.argsort(x_, kind="stable")
        return (x_[kolejnosc].astype(np.float64), y_[kolejnosc].astype(np.float64), s_[kolejnosc])
    gg = g.astype(np.float64)
    xs, ys, ss = [], [], []
    for x in range(step // 2, w, step):
        a, b = _okno_kolumny(x, w, korytarz, y0, y1, h)   # patrz _arc_trace_edge
        n = b - a
        if n < 20:
            continue
        n1 = np.arange(1, n, dtype=np.float64)
        n2 = n - n1
        col = gg[a:b, max(0, x - step // 2):x + step // 2 + 1].mean(axis=1)
        cs = np.cumsum(col)
        m1 = cs[:-1] / n1
        m2 = (cs[-1] - cs[:-1]) / n2
        strength = (n1 * n2 / float(n * n)) * (m1 - m2) ** 2
        i = int(np.argmax(strength))
        xs.append(float(x)); ys.append(float(a + i + 1)); ss.append(float(strength[i]))
    if len(xs) < 12:
        return None
    return np.array(xs), np.array(ys), np.array(ss)


# ⚠️ WYRAZ CZWARTEGO STOPNIA -- ZMIERZONY I ODRZUCONY. Nie dodawaj go bez ponownej
# kalibracji `arc_curvature`, bo policzysz ten sam efekt DWA RAZY.
#
# Horyzont na rybim oku faktycznie NIE jest parabola. Na 30 recznie narysowanych liniach
# (~/keya-horyzont-recznie) dopasowanie A + B*d + C*d^2 + E*d^4 ze SWOBODNYM C zmniejsza
# reszte z 3,26 px do 0,56 px, a wyraz E ma mediane 1,76e-11, czyli 47 px na |d|=1280.
#
# ALE: przy swobodnym dopasowaniu wychodzi C = 1,31e-4, a `arc_curvature` wymusza 1,50e-4
# -- o 13% wiecej, co na krawedzi daje ~31 px. Czyli `arc_curvature` NIE jest czysta
# krzywizna drugiego stopnia; jest krzywizna EFEKTYWNA, skalibrowana tak, by parabola
# trafiala w prawdziwy horyzont czwartego rzedu na szerokosci tego kadru. Dolozenie E
# liczy ten sam efekt ponownie.
#
# Zmierzone na tych samych 30 liniach, blad estymatora (nie ksztaltu):
#     bez d^4   mediana 13,4 px, 13/30 ponizej 10 px
#     z  d^4    mediana 30,5 px,  8/30 ponizej 10 px   -- lepiej na 12 klatkach, GORZEJ na 18
# Psuje te kadry, ktore byly dobre (2,4 -> 36,7 px; 36,9 -> 142,2), a poprawia o kilka px
# te, ktore i tak sa zepsute o 150+.
#
# Zrobienie tego porzadnie wymaga PRZEKALIBROWANIA C i E RAZEM na recznych liniach.
# Zysk na ksztalcie to ~2 px mediany wobec 13 px bledu estymatora zdominowanego przez
# slad, wiec priorytet ma slad, nie model.


def _arc_trace_step(g, y0, y1, kolumn=64, sigma=40.0):
    """Per column: row of the strongest DROP in the heavily smoothed profile.

    Why a third trace. In haze the horizon is not an EDGE, it is the BORDER BETWEEN TWO
    REGIONS -- and waves and sun glitter give locally stronger gradients than the sky/water
    boundary itself, so the edge trace locks onto water and the arc lands hundreds of px
    low. Row profiles show it plainly: the gradient curve has as many peaks in the water as
    at the horizon, while the luminance curve has a clear KNEE there.

    Smoothing vertically with sigma=40 rows kills the swell (period 5-20 rows) and keeps
    the step (50-100 rows). This is deliberately NOT a replacement for the edge trace --
    ★ LICZBY PONIZEJ POPRAWIONE 2026-09-01. Pierwotnie stalo tu "24,5% -> 2,2%, 11x lepiej",
    zmierzone kryterium "box nad horyzontem" -- ktore jest JEDNOSTRONNE: widzi tylko luk
    za nisko, wiec nagradzalo ten slad takze wtedy, gdy przesunal luk ZA WYSOKO. Na 30
    RECZNIE narysowanych liniach horyzontu (~/keya-horyzont-recznie, patrz pamiec projektu)
    prawdziwy obraz jest skromniejszy -- mediana bledu wobec reki:

        gradient + Otsu     13,4 px   13/30 ponizej 10 px    8/30 powyzej 100 px
        stopien luminancji   9,0 px   16/30 ponizej 10 px    3/30 powyzej 100 px

    Per klatka to niemal remis (lepiej 10, gorzej 9, remis 11). Realny zysk siedzi w
    KATASTROFACH: 8 -> 3 na 30. Dlatego to zapas, a nie zamiennik -- wlacza sie tylko,
    gdy pozostale slady nie zmieszcza sie w `max_res`.

    Skala operatora jest tu istota rzeczy: `_arc_trace_edge` liczy roznice srednich
    12 wierszy nad i pod (okno 24 px), wiec fala o okresie 5-20 wierszy jest dla niego
    sygnalem. Ten slad calkuje ja gaussem o sigma 40 (jadro 241 wierszy) i dopiero na tym
    liczy pochodna.
    """
    h = g.shape[0]
    y0 = max(0, int(y0)); y1 = min(h, int(y1))
    if y1 - y0 < 40:
        return None
    sub = g[y0:y1]
    # Sigma MUSI byc mala wobec okna: przy jadrze porownywalnym z wysokoscia okna pik
    # spadku zawsze wypada na jego krawedzi i slad odrzuca wszystkie kolumny.
    sigma = float(min(sigma, max(4.0, (y1 - y0) / 8.0)))
    k = int(sigma * 6) | 1
    # 64 kolumny, nie staly krok: `_arc_fit` odrzuca najslabsza cwiartke wag i wymaga
    # potem >=20 punktow, wiec slad musi ich dostarczyc z zapasem. Ta sama liczba, na
    # ktorej zmierzono 24,5% -> 2,2%.
    kraw = np.linspace(0, g.shape[1], int(kolumn) + 1).astype(int)
    xs, ys, ss = [], [], []
    for i_kol in range(int(kolumn)):
        a, b = kraw[i_kol], kraw[i_kol + 1]
        if b <= a:
            continue
        col = sub[:, a:b].mean(axis=1)
        gl = cv2.GaussianBlur(col.reshape(-1, 1).astype(np.float32), (1, k), sigma).ravel()
        d = -np.gradient(gl)                      # brightness DROP: sky -> water
        i = int(np.argmax(d))
        if i <= 2 or i >= len(d) - 3:
            continue                              # a peak on the window edge is not a step
        xs.append((a + b) / 2.0); ys.append(float(y0 + i)); ss.append(float(d[i]))
    if len(xs) < 8:
        return None
    ss = np.clip(np.array(ss), 0, None)
    if ss.max() <= 0:
        return None
    return np.array(xs), np.array(ys), ss / ss.max()


# ⚠️ WYNIK UJEMNY -- `b_fixed` ISTNIEJE, ALE NIE UZYWAJ GO BEZ NOWEGO POMIARU.
# Wiazanie B przechylem POGARSZA luk: zmierzone na 400 klatkach kryterium boxow
# PRZY KRAWEDZIACH kadru (tam obrot boli najbardziej), 10,2% -> 19,6% niemozliwych
# polozen. Powod: ponizsza zaleznosc jest POPULACYJNA, nie per klatka -- rozrzut
# wokol niej (mediana 0,91 st., p90 3,56) jest wiekszy niz blad, ktory wlasne
# dopasowanie osiaga samo. Wymuszenie modelu wyrzuca lepszy pomiar za gorsza srednia.
# POTWIERDZONE 2026-09-02 na 46 kadrach RECZNEJ prawdy (0,4-28,9 st. przechylu obrazu,
# 30 kadrow `~/keya-horyzont-recznie` + 16 `~/keya-horyzont-przechyl`): wiez daje mediane
# 164,5 px wobec 10,8 px bez niej, 29/46 kadrow powyzej 100 px. Mechanizm juz widac:
# czujnik przewiduje kat DOBRZE, ale wymuszenie go zostawia A jedynym stopniem swobody,
# wiec dopasowanie idzie za falami i wypycha A o kilkaset px.
# BRAMKA proponowana ponizej ZMIERZONA I ODRZUCONA: na 6 kadrach ocenionych przez
# uzytkownika jako katastrofalne mediana odchylki od przewidywania wynosi 11,4 st.,
# a na 10 poprawnych 7,7 st. -- rozklady nachodza, przy kazdym progu tracisz tyle
# kontroli, ile lapiesz katastrof.
# Przechyl przydaje sie natomiast do KRZYWIZNY -- patrz arc_curvature(heel_deg=).
#
# Kat luku wobec PRZECHYLU z AHRS, dopasowany na 19 515 klatkach archiwum:
#     kat_luku [st.] = -0,807 * przechyl [st.] - 0,55      korelacja -0,889
# Wspolczynnik NIE wynosi -1: luk niedokreca sie o okolo 20%, systematycznie i tym
# mocniej, im bardziej lodz lezy (przy 20 st. przechylu to 4 st. bledu). Nie wiadomo,
# czy to blad dopasowania, czy fizyka masztu -- kamera wisi 19,2 m nad woda, wiec jej
# przechyl moze realnie byc mniejszy niz kadluba. Dla wiezu to bez roznicy: liczy sie
# ZMIERZONA odpowiedz, nie ta zakladana.
# ⚠️ NIE ZMIENIAJ -0,807 NA -0,978. Dopasowanie kata na calym dniu z duzym przechylem
# daje -0,978 (korelacja -0,919), ale to liczba ZANIECZYSZCZONA przez te same uciekajace
# dopasowania, ktore mial opisywac. Na recznej prawdzie (46 kadrow) mediana bledu
# przewidywania kata wynosi 1,05 st. dla -0,807, 1,89 st. dla -0,978 i 1,30 st. dla
# swobodnego dopasowania do prawdy (-0,866). Stala z repo jest najlepsza z trzech.
_HEEL_DO_KATA_A = -0.807
_HEEL_DO_KATA_B = -0.55


def heel_to_slope(heel_deg):
    """Wyraz B luku (dy/dx w srodku kadru) odpowiadajacy przechylowi, albo None."""
    if heel_deg is None:
        return None
    try:
        return math.tan(math.radians(_HEEL_DO_KATA_A * float(heel_deg) + _HEEL_DO_KATA_B))
    except (TypeError, ValueError):
        return None


def arc_visible_range(a_row, heel_deg, frame_w, frame_h):
    """(x0, x1) -- kolumny, w ktorych luk o wierszu `a_row` na SRODKU kadru jest jeszcze
    w kadrze, albo None gdy przechylu nie znamy albo luk nie wychodzi wcale.

    Przy polozonej lodzi horyzont wychodzi GORNA krawedzia mniej wiecej w polowie
    szerokosci -- na 16 recznie obrysowanych kadrach (przechyl 21-31 st.) konczyl sie
    miedzy 1267 a 1657 px z 2560. Dalej horyzontu w kadrze NIE MA, a sa tam za to
    najsilniejsze gradienty blasku, ktore luk przeglosowuja. Przewidywanie z przechylu
    myli sie asymetrycznie i w bezpieczna strone: krawedz od strony nawietrznej co do
    piksela (mediana -2 px, p90 3), przeciwna ZA DALEKO (mediana +228 px), wiec okno
    nigdy nie utnie prawdziwego horyzontu, tylko utnie mniej, niz mogloby.

    ⚠️ PRZECHYL MA ZNAK. Na lewy burt luk wychodzi PRAWA krawedzia, na prawy LEWA --
    dlatego zwracamy zakres obustronny, a nie sama gorna granice. Pierwsza wersja
    zwracala `max(widoczne)` i przy ujemnym przechyle po cichu nie robila nic.

    Zmierzone na pelnej sciezce, 46 kadrow recznej prawdy (z `hint_row`, czyli tak jak
    dziala na lodzi): przy duzym przechyle mediana bledu 176,5 -> 8,3 px, kadrow ponizej
    20 px 0 -> 11; lacznie 62,6 -> 8,3 px i 19 -> 32. Dziala tez z podpowiedzia pomylona
    o 120 px. Marginesu NIE dodawac -- juz 100 px w pionie rozszerza okno z 72% na 97%
    szerokosci i caly zysk znika.
    """
    b = heel_to_slope(heel_deg)
    if b is None:
        return None
    c = arc_curvature(float(a_row), frame_w, heel_deg)
    x = np.arange(0.0, float(frame_w), 1.0)
    d = x - float(frame_w) / 2.0
    y = float(a_row) + b * d + c * d * d
    w = np.flatnonzero((y >= 0.0) & (y < float(frame_h)))
    if len(w) == 0:
        return None
    x0, x1 = int(w.min()), int(w.max()) + 1
    if x0 <= 0 and x1 >= frame_w:
        return None                                  # luk nie wychodzi z kadru
    return x0, x1


def _arc_fit(x, y, w, frame_w, iters=8, b_fixed=None, heel_deg=None):
    """Robustly fit A and B with C tied to A. Returns (A, B, C, residual) or None.

    Iteratively reweighted: clouds, vessels and wave crests get squeezed out. Only
    two free parameters, because the third is not ours to guess (see arc_curvature).

    `b_fixed` ties B as well -- to the measured heel, the same way C is tied to
    curvature. Then only A is free. Use it when attitude is available and trusted;
    a free B drifts with whatever texture the trace latched onto.
    """
    keep = w > np.percentile(w, 25)
    if keep.sum() < 20:
        return None
    d = x[keep] - frame_w / 2.0
    yy = y[keep]
    ww = w[keep].astype(np.float64)
    c = arc_curvature(float(np.median(yy)), frame_w, heel_deg)
    k = r = None
    for _ in range(iters):
        target = yy - c * d * d
        if b_fixed is None:
            design = np.vstack([np.ones_like(d), d]).T
        else:
            target = target - float(b_fixed) * d      # B zwiazane -- zostaje samo A
            design = np.ones((len(d), 1))
        k = np.linalg.lstsq(design * ww[:, None], target * ww, rcond=None)[0]
        r = target - design @ k
        if b_fixed is not None:
            k = np.array([k[0], float(b_fixed)])
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        ww = ww * np.exp(-0.5 * (r / (2.0 * s)) ** 2)
        if ww.sum() < 1e-9:
            return None
        c = arc_curvature(k[0], frame_w, heel_deg)
    return float(k[0]), float(k[1]), float(c), float(np.median(np.abs(r)))


def horizon_arc_candidates(frame: np.ndarray, hint_row=None) -> list:
    """Independent horizon-arc estimates for one frame, best-residual first.

    Two traces (edge and Otsu) over a band around `hint_row` when we have one --
    the previous frame's arc, the mounting calibration, anything. Without a hint
    the band is the top 60% of the frame, which is where a horizon can be.

    Returns a list of (A, B, C, residual). NEITHER estimator wins everywhere --
    measured per-job scatter against hand-drawn points was 9/224/172 px for the
    edge trace and 36/25/49 px for Otsu on the same three jobs -- so the caller
    picks between them, ideally by agreement with recent frames (the horizon does
    not jump), which is what the watchkeeper does.
    """
    if frame is None or frame.ndim != 3:
        return []
    h, w = frame.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (9, 5), 0)
    sk = w / _ARC_REF_W                                # bands were measured at 2560px
    if hint_row is None:
        bands = [(0, int(h * 0.60))]
    else:
        bands = [(max(0, hint_row - 170 * sk), min(h, hint_row + 330 * sk))]
    out = []
    for y0, y1 in bands:
        for trace in (_arc_trace_edge, _arc_trace_otsu):
            t = trace(g, y0, y1)
            if t is None:
                continue
            fit = _arc_fit(t[0], t[1], t[2], w)
            if fit is not None:
                out.append(fit)
    out.sort(key=lambda p: p[3])
    return out


def horizon_arc_tiles(frame: np.ndarray, hint_row=None, ranges=None,
                      max_res=None, heel_deg=None) -> list:
    """Jeden luk NA KAFEL, z jednego sladu na caly kadr. -> [[kandydaci], ...] per kafel.

    Po co, skoro jest juz `horizon_arc_candidates`: JEDEN kwadrat nie opisuje horyzontu
    na calej szerokosci 2560 px. Zmierzone na zamrozonym zbiorze (4727 klatek, podzial
    po dniach, ocena na KOLUMNACH TRZYMANYCH Z BOKU, wiec czterokrotnie wieksza liczba
    parametrow nie wygrywa sama z siebie): blad wiersza 12,3 -> 5,5 px na `train` i
    12,2 -> 7,1 px na `test`. Roznica utrzymuje sie przy obu parametryzacjach C, wiec
    to wlasnosc modelu, nie sposobu dopasowania.

    Slad liczymy RAZ na caly kadr i tylko DZIELIMY go na zakresy kolumn -- droga czesc
    (dwa przebiegi po pikselach) nie zwielokrotnia sie, dochodzi tylko kilka `lstsq` na
    ~80 punktach zamiast jednego na 320.

    `frame_w` w `_arc_fit` zostaje szerokoscia CALEGO kadru, bo krzywizna jest wlasnoscia
    obiektywu i kadru, a nie wycinka: przekazanie tu szerokosci kafla zawyzyloby C 3,2x
    (patrz arc_curvature -- skaluje sie odwrotnie do szerokosci).
    """
    if frame is None or frame.ndim != 3:
        return []
    h, w = frame.shape[:2]
    if ranges is None:
        ranges = [(0, w)]
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (9, 5), 0)
    sk = w / _ARC_REF_W
    if hint_row is None:
        y0, y1 = 0, int(h * 0.60)
    else:
        y0, y1 = hint_row - 170 * sk, hint_row + 330 * sk
        # Pas -170..+330 opisuje luk POZIOMY. Przy przechyle luk rozciaga sie w pionie
        # o |B|*w, czyli przy 25 st. o ponad 1100 px na kadrze 2560 -- pas obcinal wtedy
        # niemal caly horyzont i dopasowanie mialo do dyspozycji same fale. Zmierzone na
        # 16 recznych obrysach: mediana bledu 142,2 px z waskim pasem, 5,8 px z szerokim.
        b = heel_to_slope(heel_deg)
        if b is not None:
            d = np.array([-w / 2.0, w / 2.0])
            yy = hint_row + b * d + arc_curvature(hint_row, w, heel_deg) * d * d
            y0, y1 = min(y0, yy.min() - 60 * sk), max(y1, yy.max() + 60 * sk)
        y0, y1 = max(0, int(y0)), min(h, int(y1))
    # KORYTARZ wzdluz przewidzianego luku zamiast wspolnego pasa poziomego -- patrz
    # _arc_trace_edge. Dwa przebiegi: zgrubny luk z podpowiedzi, potem korytarz wokol
    # NIEGO. Drugi przebieg jest konieczny (mediana 8,4 -> 4,8 px), trzeci juz nic nie
    # daje. Bez podpowiedzi korytarza nie ma i zostaje pas jak dotad.
    korytarz = None
    if hint_row is not None:
        b_kor = heel_to_slope(heel_deg) or 0.0
        korytarz = ((float(hint_row), b_kor, arc_curvature(hint_row, w, heel_deg)),
                    _KORYTARZ_POLOWA * sk)
        wstepny = _arc_trace_edge(g, y0, y1, korytarz=korytarz)
        if wstepny is not None:
            f = _arc_fit(wstepny[0], wstepny[1], wstepny[2], w, heel_deg=heel_deg)
            if f is not None:
                korytarz = ((f[0], f[1], f[2]), _KORYTARZ_POLOWA * sk)
    slady = [t for t in (_arc_trace_edge(g, y0, y1, korytarz=korytarz),
                         _arc_trace_otsu(g, y0, y1, korytarz=korytarz)) if t is not None]
    # Za miejscem, w ktorym luk wychodzi gorna krawedzia, horyzontu w kadrze NIE MA --
    # patrz arc_visible_range. Odcinamy tam punkty sladu, zamiast pozwolic blaskowi
    # przeglosowac horyzont. Zakres jest OBUSTRONNY, bo przechyl ma znak: na jeden burt
    # luk wychodzi prawa krawedzia, na drugi lewa. Wymaga podpowiedzi (poprzednia klatka)
    # i przechylu; bez ktoregokolwiek z nich zostaje caly kadr.
    if hint_row is not None:
        widok = arc_visible_range(hint_row, heel_deg, w, h)
        if widok is not None:
            x0w, x1w = widok
            przyciete = []
            for sx, sy, ss in slady:
                m = (sx >= x0w) & (sx < x1w)
                przyciete.append((sx[m], sy[m], ss[m]) if m.sum() >= 20 else (sx, sy, ss))
            slady = przyciete
    slad_stopnia = None                     # liczony leniwie, tylko gdy powyzsze zawioda
    # Przechyl poprawia KRZYWIZNE (arc_curvature), a NIE wiaze kata. Wiez B zmierzona
    # trzykrotnie i odrzucona -- na 46 kadrach recznej prawdy daje mediane 164,5 px
    # wobec 10,8 px bez niej i 29/46 kadrow powyzej 100 px. Powod: czujnik przewiduje
    # kat dobrze (1,05 st.), ale wymuszenie go wypycha A o kilkaset px, bo slad ma przy
    # blasku najsilniejsze poparcie na falach nisko w kadrze, a A jest wtedy jedynym
    # stopniem swobody, ktorym dopasowanie moze za nimi pojsc.
    out = []
    for x0, x1 in ranges:
        kand = []
        for x, y, s in slady:
            m = (x >= x0) & (x < x1)
            if m.sum() < 20:
                continue
            fit = _arc_fit(x[m], y[m], s[m], w, heel_deg=heel_deg)
            if fit is not None:
                kand.append(fit)
        # ZAPAS: gdy zaden slad nie trafil ponizej progu, sprobuj stopnia luminancji.
        # Liczony leniwie -- kosztuje tylko wtedy, gdy poprzednie zawiodly. Patrz
        # _arc_trace_step: 24,5% -> 2,2% na kadrach, gdzie krawedz zawodzi.
        if max_res is not None and (not kand or kand[0][3] > max_res):
            if slad_stopnia is None:
                slad_stopnia = _arc_trace_step(g, y0, y1)
            if slad_stopnia is not None:
                sx, sy, ss = slad_stopnia
                m = (sx >= x0) & (sx < x1)
                if m.sum() >= 8:
                    fit = _arc_fit(sx[m], sy[m], ss[m], w, heel_deg=heel_deg)
                    if fit is not None:
                        kand.append(fit)
        kand.sort(key=lambda p: p[3])
        out.append(kand)
    return out


# Refracted Earth radius: k=0.13 over water. Plain 6371 km moves the numbers by <10%
# and the wrong direction -- measured, see ais_waterline_row.
_R_EFF_M = 7.2e6


def horizon_dip_range_m(cam_height_m: float = 19.2, r_eff_m: float = _R_EFF_M) -> float:
    """Range at which a waterline sits exactly ON the horizon: sqrt(2*R*h). ~16.6 km at 19.2 m."""
    return math.sqrt(2.0 * r_eff_m * max(0.1, float(cam_height_m)))


def ais_waterline_depression(range_m: float, cam_height_m: float = 19.2,
                             focal_px: float = 1300.0, r_eff_m: float = _R_EFF_M) -> float:
    """Pixels BELOW the horizon where a target's waterline sits, at a known range.

    NOT focal*atan(h/range) -- that formula ignores the horizon's own dip and overstates
    the depression by ~3x at long range (it gives 3.6 px at 6933 m where the geometry
    gives 1.2). Correct form, measured against 1049 hand-drawn boxes with AIS ranges:

        depression = focal * (d - d_h)^2 / (2 * R * d),   d_h = sqrt(2*R*h)

    and 0 beyond d_h, where the hull is over the horizon and only superstructure shows.
    Values at h=19.2 m: 500 m -> 47 px | 1000 -> 22 | 2000 -> 9.7 | 5000 -> 2.4 |
    7000 -> 1.2 | 10000 -> 0.4 | >16.6 km -> 0.

    Rejected on measurement: raising the camera height to close the gap (it would need
    103 m; the fit's own minimum sits at 19-25 m), and dropping refraction (k=0 makes it
    worse, 1.12 -> 1.22 px at 6933 m).
    """
    d = float(range_m or 0.0)
    d_h = horizon_dip_range_m(cam_height_m, r_eff_m)
    if d <= 0.0 or d >= d_h:
        return 0.0
    return focal_px * (d - d_h) ** 2 / (2.0 * r_eff_m * d)


# Curvature bias of the fitted arc, px per u^2 with u = (column - frame_w/2)/1000.
# The arc's C comes out ~7% low, so boxes near the frame edge sit ~9.5 px BELOW the
# predicted row at |column - centre| > 900 while the centre is unbiased. Measured on the
# annotation side (cvat-keya/kal_zanurzenie.POPR_KRZYWIZNY) against hand-drawn boxes with
# AIS ranges; the same arc model is used here, so the same bias applies. It is NOT part of
# the waterline geometry -- it corrects the arc, which is why it is added to the arc row
# and not to the depression.
ARC_CURVATURE_BIAS_PX = 10.4


def ais_waterline_row(arc, column: float, frame_w: int, range_m: float,
                      cam_height_m: float = 19.2, arc_bias_px: float = 0.0,
                      curvature_bias: bool = True) -> float:
    """Image row where a sensor target's WATERLINE should appear. Needs an arc.

    arc_bias_px is a per-session correction for the arc itself, which is the dominant
    error: across 18 recordings the fitted arc sits anywhere from -15 to +26 px off the
    true waterline reference (sd 11.4 px), and nothing per-frame predicts it -- residual
    vs the fit residual, vs A, vs heel, vs pitch, vs column all give |spearman| <= 0.11.
    Calibrate it from confirmed detection/target pairs, or leave 0 and widen the gate.
    Watchkeeper._arc_bias does exactly that on the boat; the annotation side calls the
    equivalent kal_zanurzenie.kalibruj (measured: 10 px error without it, 4.3 px from one
    anchor, 2.8 px from ten).

    curvature_bias adds ARC_CURVATURE_BIAS_PX -- a property of the arc FIT, not of the
    target, so it applies whether or not arc_bias_px is known. Off only for callers that
    already corrected the arc themselves.
    """
    d = float(column) - frame_w / 2.0
    popr = ARC_CURVATURE_BIAS_PX * (d / 1000.0) ** 2 if curvature_bias else 0.0
    return (horizon_arc_row(arc, column, frame_w) + popr
            + ais_waterline_depression(range_m, cam_height_m)
            + float(arc_bias_px))


def horizon_arc_row(arc, column: float, frame_w: int) -> float:
    """Horizon row at `column` for an arc from horizon_arc_candidates()."""
    d = float(column) - frame_w / 2.0
    return arc[0] + arc[1] * d + arc[2] * d * d


def estimate_horizon(frame: np.ndarray, band: tuple = (0.02, 0.96)) -> tuple[int, float]:
    """Horizon estimate with a CONFIDENCE the caller must respect: (y, conf in [0,1]).

    y is the fitted line's row at the CENTRE column; the line's tilt is dropped here.
    Callers that need the tilt (roll) use estimate_horizon_line -- same estimator.
    """
    y_c, _slope, conf = estimate_horizon_line(frame, band)
    return int(round(y_c)), conf


def estimate_horizon_line(frame: np.ndarray, band: tuple = (0.02, 0.96)) -> tuple[float, float, float]:
    """Horizon LINE: (row at the centre column, slope dy/dx, conf in [0,1]).

    The row at column x is  y_c + slope * (x - W/2). Same estimator and confidence as
    estimate_horizon, which only drops the slope; a level row is wrong by slope*W/2 at the
    frame edges (~33 px at 4 deg on 960 px), which matters wherever geometry is measured
    from the horizon. Without a line the fallback is (0.55*H, 0.0, 0.0).

    Per-column strongest vertical-gradient candidate -> RANSAC line fit ->
    verification (inlier support x sky-smoothness above the line x edge contrast).
    Low conf means "no trustworthy horizon in this frame" (haze, glare, camera
    pitched off the line, all-water view) -> the caller must FAIL OPEN and skip
    horizon filtering; guessing a line and filtering on it is the failure mode
    that deletes real boats.

    Replaces estimate_horizon_texture, whose "~10x more stable" claim did not
    survive ground truth: on 4,643 human-labeled horizon lines (ship-mounted +
    boat-mounted footage, two independent sources) texture had median error
    ~19%H and put the line >8%H too HIGH (over-filtering direction) on 18% of
    frames. This estimator, gated at conf>=0.30, has median error 0.3-0.8%H
    with dangerous-direction errors ~1-2%, all isolated single frames that the
    caller's temporal smoothing absorbs (validated on a by-video holdout).

    Uniform rows (letterbox bars / UI banners) are excluded from the search --
    their razor-straight edge is a fake horizon with a perfectly smooth "sky".
    ~3-5 ms at 480px working width; pure cv2+numpy.
    """
    H0, W0 = frame.shape[:2]
    W = 480
    scale = W / W0
    h = max(1, int(round(H0 * scale)))
    g0 = cv2.cvtColor(cv2.resize(frame, (W, h)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g0, (5, 5), 0)
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    fallback = int(H0 * 0.55)

    # uniformity test on the UNBLURRED gray: letterbox bars are digitally constant
    # (std ~0) while any real content -- even flat calm sky -- carries sensor noise.
    row_std = g0.std(axis=1)
    content = row_std > 0.5
    c_top = int(np.argmax(content)) if content.any() else 0
    c_bot = int(h - np.argmax(content[::-1])) if content.any() else h
    # margin past the content edge: the Gaussian blur smears a bar edge a few rows
    # into the content, and that smeared tail can outcompete a real (weaker) seam
    m = max(6, int(0.025 * h))
    lo, hi = max(int(band[0] * h), c_top + m), min(int(band[1] * h), c_bot - m)
    if hi - lo < 8:
        return float(fallback), 0.0, 0.0

    roi = gy[lo:hi]
    ys = roi.argmax(0) + lo                       # per-column candidate row
    strength = roi.max(0)
    xs = np.arange(W, dtype=np.float32)

    # RANSAC line fit, hypotheses drawn from the strongest candidates
    cand = np.argsort(-strength)[: max(40, W // 3)]
    best_inl = None
    rng = np.random.default_rng(0)                # fixed seed: deterministic per frame
    tol = max(2.0, 0.015 * h)
    for _ in range(60):
        i, j = rng.choice(cand, 2, replace=False)
        if xs[i] == xs[j]:
            continue
        a = (ys[j] - ys[i]) / (xs[j] - xs[i])
        if abs(a) > 0.30:                         # >~17 deg tilt: not a horizon
            continue
        b = ys[i] - a * xs[i]
        inl = np.abs(a * xs + b - ys) < tol
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
    if best_inl is None or best_inl.sum() < 0.15 * W:
        return float(fallback), 0.0, 0.0
    a, b = np.polyfit(xs[best_inl], ys[best_inl], 1)
    y_c = a * (W / 2.0) + b
    support = float(best_inl.mean())

    # verification: sky above the line must be smoother than the sea below it
    yl = np.clip((a * xs + b).astype(int), 1, h - 2)
    off = np.maximum(3, int(0.06 * h))
    up_idx = np.maximum(yl - off, c_top)
    dn_idx = np.minimum(yl + off, max(c_bot - 1, c_top + 1))
    tex = cv2.GaussianBlur(np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)) +
                           np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)), (9, 9), 0)
    cols = np.arange(W)
    above = float(np.median(tex[up_idx, cols]))
    below = float(np.median(tex[dn_idx, cols]))
    sky_term = float(np.clip((below / (above + 1e-3) - 1.0) / 3.0, 0.0, 1.0))

    line_strength = float(np.median(strength[best_inl]))
    glob = float(np.median(gy[lo:hi])) + 1e-3
    contrast = float(np.clip((line_strength / glob - 1.0) / 6.0, 0.0, 1.0))

    conf = support * (0.60 * sky_term + 0.40 * contrast)
    if y_c < 0.03 * h or y_c > 0.93 * h:          # near frame edge: may be out of view
        conf *= 0.5
    # slope is scale-invariant (rows and columns are resized by the same factor)
    return float(y_c / scale), float(a), float(np.clip(conf, 0.0, 1.0))


def _ios(a, b) -> float:
    """Intersection-over-smaller: high when one box is contained in the other."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (min(sa, sb) + 1e-9)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_tiled(detections: list[dict], contain_thr: float = 0.7,
                iou_thr: float = 0.35) -> list[dict]:
    """Merge detections from overlapping passes (full frame + band tiles).

    Clusters by CONTAINMENT (IoS), not IoU: a tall mast-box and a short hull-box of
    one boat have low IoU but high IoS -> merged; two adjacent boats have low IoS ->
    kept apart. Each cluster collapses to ONE confidence-weighted-average box (stable
    frame-to-frame so ByteTrack keeps the id); highest-conf member sets the class.
    """
    dets = sorted(detections, key=lambda d: -d["confidence"])
    clusters: list[list[dict]] = []
    for d in dets:
        for cl in clusters:
            # Against ANY member, not just the first. Comparing only with cl[0] made the
            # clustering non-transitive: with A overlapping B and B overlapping C but A not
            # reaching C, C opened a NEW cluster and one hull kept several boxes. On a target
            # found by the full-frame pass and by two band tiles that is exactly what happens,
            # and the overlay showed a pile of brackets stacked on a single boat.
            #
            # IoU as well as containment. Containment alone misses the common tiling case: two
            # boxes of similar size offset by part of their width are plainly the same object,
            # but neither is 70% inside the other.
            if any(_ios(m["bbox"], d["bbox"]) > contain_thr or _iou(m["bbox"], d["bbox"]) > iou_thr
                   for m in cl):
                cl.append(d)
                break
        else:
            clusters.append([d])
    out = []
    for cl in clusters:
        sw = sum(m["confidence"] for m in cl)
        fb = tuple(sum(m["bbox"][i] * m["confidence"] for m in cl) / sw for i in range(4))
        out.append({"bbox": fb, "confidence": cl[0]["confidence"],
                    "class": cl[0]["class"], "class_id": cl[0].get("class_id")})
    return out


def apply_horizon_mask(frame: np.ndarray, horizon_y: int, margin: int = 20) -> np.ndarray:
    """Create a copy of frame with everything below horizon+margin darkened.
    This suppresses wave texture that YOLO might interpret as objects.
    Does NOT modify the original frame.

    Instead of zeroing pixels (which could confuse the model), we apply a
    gentle suppression: reduce brightness below horizon to reduce false positive
    triggers from wave patterns.
    """
    result = frame.copy()
    h = frame.shape[0]
    mask_start = min(horizon_y + margin, h)

    # Gradual fade: 100% brightness at horizon → 60% at bottom
    for y in range(mask_start, h):
        ratio = 1.0 - 0.4 * ((y - mask_start) / max(h - mask_start, 1))
        result[y] = (frame[y].astype(np.float32) * ratio).astype(np.uint8)

    return result


def horizon_margin_px(cam_height_m: float = 2.5, focal_px: float = 1300.0,
                      min_range_m: float = 300.0) -> int:
    """How far BELOW the horizon a hull may sit and still be treated as a real target.

    The 30 px default was calibrated for a 2.5 m mount. This camera is at 19.2 m, where the same
    30 px corresponds to a vessel about 830 m away -- so every non-person, sub-0.6-confidence,
    small-box detection closer than that was being deleted EVERY FRAME, in the 0-1 NM band where
    the detector actually works (measured recall 78% / 54% / 32%). The margin is a geometry
    question, not a constant: a waterline sits atan(cam_height / range) below the horizon.

    Returns the offset for a target at min_range_m, so anything from there outward is kept."""
    import math
    return int(round(math.atan(max(cam_height_m, 0.1) / max(min_range_m, 1.0)) * focal_px))


def filter_below_horizon(detections: list[dict], horizon_y: int,
                         frame_height: int, frame_width: int, margin: int = 30,
                         keep_conf: float = 0.6) -> list[dict]:
    """Drop only wave/foam false positives sitting ENTIRELY in the foreground water.

    SEA-7(A): a real vessel at the horizon STRADDLES the line — rigging/superstructure
    above, hull at the waterline — so its TOP edge reaches up to the horizon. We test
    the box vs the line (TOP edge y1), NOT its centre: the old centre test wrongly
    dropped hull-heavy / low-freeboard boats (RIB, motorboat, ship hull) whose centre
    sits below the line — the small, low, most collision-relevant ones. Keep-biased:
    precision is CPA's 3/5 job, not this filter; a dropped real vessel is the failure
    mode, a kept wave-FP only costs a cycle.
      KEEP if the box top reaches the horizon (y1 <= horizon_y + margin)  [straddler/above]
      KEEP person (MOB), conf >= keep_conf (SEA-7 high-conf bypass), or a large box (close boat)
      else (entirely foreground + small + low-conf) -> wave FP -> DROP
    """
    keep = []
    frame_area = frame_width * frame_height
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if det.get("class") == "person":                  # MOB: never filter
            keep.append(det)
            continue
        if y1 <= horizon_y + margin:                       # box top reaches the horizon -> real target
            keep.append(det)
            continue
        if det.get("confidence", 1.0) >= keep_conf:        # high-conf bypass (belt-and-suspenders)
            keep.append(det)
            continue
        if (x2 - x1) * (y2 - y1) > frame_area * 0.05:      # large box = close boat -> keep
            keep.append(det)
            continue
        # entirely below the horizon, small, low-conf -> foreground wave FP -> drop
    return keep


# ── Vision health (SEA-FUSION 3) ───────────────────────────────────────────────

# Laplacian variance on a 0-255 grey frame. Marine scenes are low-texture by nature (sky +
# water), so these sit well below typical photographic values; they are tuned to separate
# "usable maritime view" from "lens is fogged / it is dark", not to grade image quality.
# Calibrated against 42 real marine frames (32 stills + samples from the 5 MOB clips):
# sharpness min 4.9 / p10 96.6 / median 409.3, contrast min 15.2 / p10 22.5 / median 33.2.
# Sharpness discriminates (the one unusable frame measured 4.9, ~100x under the median);
# contrast does NOT -- a flat overcast sea is legitimately low-contrast while staying sharp.
VISION_BLUR_DEGRADED = 60.0    # below this the frame carries little edge detail
VISION_BLUR_BLIND = 15.0       # essentially no detail -- condensation, spray, total dark
VISION_CONTRAST_DEGRADED = 12.0  # was 18.0, which cut INTO the real distribution (floor 15.2)
VISION_BLUR_SOFT = 200.0       # contrast only implies fog when detail is ALSO down (see below)


def assess_vision_health(frame, blur_degraded=VISION_BLUR_DEGRADED,
                         blur_blind=VISION_BLUR_BLIND,
                         contrast_degraded=VISION_CONTRAST_DEGRADED,
                         blur_soft=VISION_BLUR_SOFT) -> dict:
    """Cheap per-frame check of whether the CAMERA can still see -- condensation inside the
    housing, spray, fog, darkness. Returns {sharpness, contrast, status, usable}.

    Why this exists: the fusion layer trusted a fogged lens exactly as much as a clear one, so
    a blind camera degraded silently -- the operator saw "0 detections", which is
    indistinguishable from "calm empty sea". Surfacing it lets the boat lean on AIS/radar and,
    more importantly, TELLS someone the camera needs attention.

    Deliberately NOT used to suppress detections: a partly-fogged lens still sees real boats,
    and dropping detections because the image looks bad is exactly the failure mode that
    deletes a real target. Report, don't filter."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return {"sharpness": None, "contrast": None, "status": "no_frame", "usable": False}
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    contrast = float(np.std(grey))
    if sharpness < blur_blind:
        status = "blind"
    elif sharpness < blur_degraded:
        status = "degraded"
    elif contrast < contrast_degraded and sharpness < blur_soft:
        # Washed out AND soft. Contrast alone is not evidence of fog: measured on the MOB
        # clips, flat grey sea-and-sky reads contrast 15-17 while staying sharp (287-446), and
        # the old contrast-only rule called exactly those frames degraded -- a false alarm on
        # the footage the system most has to work in. Scattering blurs edges as well as
        # flattening tone, so real fog trips both.
        status = "degraded"
    else:
        status = "ok"
    return {
        "sharpness": round(sharpness, 1),
        "contrast": round(contrast, 1),
        "status": status,
        "usable": status in ("ok", "degraded"),
    }


# ── Own-shadow rejection (N-69) ──────────────────────────────────────────────

def solar_position(lat_deg, lon_deg, when_utc):
    """Sun (azimuth deg true, elevation deg above horizon) for a position and a UTC instant.

    NOAA low-precision algorithm: good to ~0.1 deg, which is an order finer than the 15 deg
    gate it feeds. Needs no ephemeris file and no third-party package."""
    import math
    y, mth = when_utc.year, when_utc.month
    if mth <= 2:
        y -= 1
        mth += 12
    a = y // 100
    b = 2 - a + a // 4
    day = when_utc.day + (when_utc.hour + when_utc.minute / 60.0
                          + when_utc.second / 3600.0) / 24.0
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (mth + 1)) + day + b - 1524.5
    n = jd - 2451545.0
    mean_long = math.radians((280.460 + 0.9856474 * n) % 360.0)
    anomaly = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = mean_long + math.radians(1.915 * math.sin(anomaly) + 0.020 * math.sin(2 * anomaly))
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    gmst = (18.697374558 + 24.06570982441908 * n) % 24.0
    hour_angle = math.radians((gmst * 15.0 + lon_deg) % 360.0) - ra
    lat = math.radians(lat_deg)
    el = math.asin(math.sin(lat) * math.sin(dec)
                   + math.cos(lat) * math.cos(dec) * math.cos(hour_angle))
    az = math.atan2(-math.sin(hour_angle) * math.cos(dec),
                    math.cos(lat) * math.sin(dec)
                    - math.sin(lat) * math.cos(dec) * math.cos(hour_angle))
    return math.degrees(az) % 360.0, math.degrees(el)


def filter_own_shadow(detections, *, frame_h, frame_w, heading_deg, lat, lon, when_utc,
                      focal_px=1300.0, bearing_offset_deg=0.0,
                      tol_deg=15.0, base_frac=0.88, min_sun_el=3.0,
                      horizon_row=None, tip_tol_deg=5.0):
    """Drop detections that are this boat's own mast shadow on the water.

    The masthead camera looks down at the water, and with the sun 15-38 deg up and the ANTI-SOLAR
    direction inside the forward FOV, the rig lays a long dark streak across the surface that the
    detector reports as a 'sailboat' -- confidently. Measured over one 3 h passage: 1,904 such
    detections, 94% of everything reported in that window, present in 68% of its frames, peak
    confidence 0.80 (the sailboat track gate is 0.40), and it raised four CRITICAL alarms against
    a phantom 12 m away.

    The evidence that it is a shadow and not a boat is geometric and parameter-free: the shadow of
    an object at the CAMERA's own height is depressed below the horizon by exactly the sun's
    elevation. Predicted box tops of 1001/801/607 px for three consecutive hours came out at
    1040/791/584 measured, residuals under 2 deg, with no fitted constant. Own sail or rigging
    would hold a FIXED relative bearing; glitter sits 180 deg the other way; a real vessel would
    have to hold a range growing exactly as h/tan(elevation) for three hours.

    Two conditions, because either alone would cost real targets: the detection must lie within
    `tol_deg` of the anti-solar bearing AND its base must sit in the bottom `1 - base_frac` of the
    frame, i.e. an implied waterline range of roughly 20 m. On the archive that pair removed 1,945
    detections and NONE of the 2,574 observations independently explained by AIS or radar -- zero
    measured recall cost. A genuine vessel that close, dead down-sun, is also on radar.

    THIRD CONDITION, and the one that stops this deleting boats. Validated over the full season
    rather than the two calm days it was first measured on: the two conditions above dropped 8
    REAL vessels (16 detections, 12 encounters, hulls at 11-19 m implied range, confidence to
    0.88, five of them CONFIRMED tracks) -- including four during a 6.5 kn harbour transit under
    a fully overcast sky, where no shadow can exist. The failure mode is confined water with the
    sun high (median 53 deg for the bad drops vs 40 for the true shadows).

    The discriminator is the same parameter-free geometry the docstring already relies on: a
    shadow cast by something at the camera's height has its TIP exactly the sun's elevation below
    the horizon, and nothing of it reaches higher. A hull does. So reject only when the box TOP is
    at or below that tip line -- `dep_top >= sun_el - tip_tol_deg` -- which is one-sided and
    cannot cost a target that sticks up. Measured on 1,946 true shadows: median |dep_top - sun_el|
    2.9 deg, 90% within 5. Applying it takes season rejections 70 -> 52, killing 8 of the 12
    non-shadow encounters (all six on 08-01), while the shadow day itself loses 2 of 1,949
    rejections and 0 of its 150 shadow tracks. Needs `horizon_row`; without it the test is skipped
    and the old two-condition behaviour stands.

    Returns the kept detections. A missing position, a missing heading or a sun below
    `min_sun_el` degrees is a no-op -- there is no shadow to reject at night."""
    import math
    if lat is None or lon is None or heading_deg is None or not detections:
        return detections
    try:
        sun_az, sun_el = solar_position(lat, lon, when_utc)
    except Exception:
        return detections
    if sun_el < min_sun_el:
        return detections
    anti_solar = (sun_az + 180.0) % 360.0
    base_row = frame_h * base_frac
    kept = []
    for d in detections:
        box = d.get("bbox") or d.get("box")
        if not box:
            kept.append(d)
            continue
        x1, y1, x2, y2 = box[:4]
        if max(y1, y2) < base_row:                     # too high in frame to be 20 m away
            kept.append(d)
            continue
        cx = (x1 + x2) / 2.0
        bearing = (heading_deg + bearing_offset_deg
                   + math.degrees((cx - frame_w / 2.0) / focal_px)) % 360.0
        off = abs((bearing - anti_solar + 540.0) % 360.0 - 180.0)
        if off > tol_deg:
            kept.append(d)
            continue
        if horizon_row is not None:
            dep_top = math.degrees((min(y1, y2) - horizon_row) / focal_px)
            if dep_top < sun_el - tip_tol_deg:
                kept.append(d)          # reaches ABOVE the shadow tip -- a hull, not a shadow
    return kept

# --------------------------------------------------------------- koperta fizyczna
# Zakres WYSOKOSCI ramki dopuszczalny przy danym ZANURZENIU (ile px dol ramki lezy
# ponizej horyzontu w JEJ kolumnie). Zmierzone na 16 331 recznych/zweryfikowanych
# boxach z jobow COMPLETED (`~/cvat-keya/koperta_prog.py`; dopasowane na splicie train,
# ocenione na val). Monotoniczna, bo taka jest fizyka kamery na 19,2 m: cel BLISKO jest
# gleboko pod horyzontem I duzy, daleki -- tuz pod horyzontem I maly.
#
# Po co: zgodnosc z koperta rozdziela trafienia od falszywek 92,1% do 51,2%, wiec pozwala
# ZLAGODZIC bramke tam, gdzie geometria detekcje popiera. Zmierzone na val (v8bis, 2 347
# detekcji): prog 0,05 dla zgodnych i 0,25 dla reszty daje 1 047 trafien / 415 falszywek
# wobec 933 / 385 przy plaskim 0,10 -- czyli +12% detekcji przy NIEZMIENIONYM stosunku
# falszywek (0,40 vs 0,41). Ta sama logika co `horizon_band_gate`: swiadkiem jest geometria.
#
# ★ WYLACZNIE do lagodzenia progu, NIGDY do kasowania. Wariant kasujacy ma nieco lepszy
# stosunek (0,39), ale N-90 opisuje, jak filtr, ktory MOGL kasowac, usunal 8 prawdziwych
# statkow przez sezon. Do tego dol ramki nie zawsze jest linia wody (zaslonięty kadlub,
# ramka przycieta krawedzia kafla) -- wtedy koperta dostaje zly argument, a bramka
# jednostronna moze wtedy najwyzej nie pomoc.
#
# Rozbicie PER KLASA sprawdzone i ODRZUCONE: 1 044 trafien / 412 falszywek wobec
# 1 047 / 415 dla wspolnej, czyli roznica trzech sztuk. Powod: koperte trzeba wybrac
# KLASA DETEKTORA, a ta jest w tym projekcie szumem (ten sam maszt bywa sailboat,
# fishing_net i buoy), wiec dzielenie danych po szumie nie dokłada informacji.
# ★ ULAMKI WYSOKOSCI KADRU, nie piksele -- ta sama pulapka, co px_frac w horizon_band_gate.
# Pomiar powstal na kadrze 1440 wierszy, wiec dzielimy przez 1440. Stala pikselowa znaczylaby
# co innego na kadrze 240 wierszy (gdzie 50 px to 21% wysokosci) niz na 1440 (3,5%) -- i pierwsza
# wersja tej koperty faktycznie przepuszczala ogromny obiekt na malym kadrze jako "zgodny".
_H_POMIARU = 1440.0
KOPERTA = tuple(
    (a / _H_POMIARU, b / _H_POMIARU, lo / _H_POMIARU, hi / _H_POMIARU)
    for a, b, lo, hi in (
        (-20.0, 5.0, 8.6, 20.7), (5.0, 8.0, 9.5, 22.7), (8.0, 12.0, 9.0, 31.1),
        (12.0, 15.0, 9.5, 33.0), (15.0, 18.0, 10.2, 35.6), (18.0, 24.0, 9.8, 41.8),
        (24.0, 35.0, 10.4, 58.2), (35.0, 240.0, 22.5, 123.8)))


def koperta_zgodna(zanurzenie, wysokosc, frame_h):
    """Czy para (zanurzenie, wysokosc ramki) miesci sie w zmierzonej kopercie fizycznej.

    Wszystkie trzy argumenty w PIKSELACH tego kadru; przeliczenie na ulamki robi ta
    funkcja, zeby wolajacy nie musial pamietac o skali.

    Poza zakresem pomiaru zwraca False -- to znaczy "brak poparcia", nie "odrzuc".
    Wolajacy uzywa tego wylacznie do OBNIZENIA bramki, wiec False = bramka bez zmian.
    """
    if not frame_h:
        return False
    z, w = zanurzenie / float(frame_h), wysokosc / float(frame_h)
    for a, b, lo, hi in KOPERTA:
        if a <= z < b:
            return lo <= w <= hi
    return False
