#!/usr/bin/env python3
"""Case-comparison metrics: eval_metrics.py <reference.png> <render.png> [--json out]
Full set of computable metrics; automatically skip metrics with missing dependencies and mark them null. All scores are for reporting only, never gates.
"""
import argparse, json, math, sys
import numpy as np
from PIL import Image


def load(p, size=None):
    im = Image.open(p).convert('RGB')
    if size: im = im.resize(size, Image.LANCZOS)
    return np.asarray(im).astype(np.float64) / 255.0


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def tile_scores(a, b, fn, grid=4):
    H, W = a.shape[:2]
    vals = []
    for i in range(grid):
        for j in range(grid):
            sa = a[i*H//grid:(i+1)*H//grid, j*W//grid:(j+1)*W//grid]
            sb = b[i*H//grid:(i+1)*H//grid, j*W//grid:(j+1)*W//grid]
            vals.append(fn(sa, sb))
    return float(np.mean(vals)), float(np.min(vals))


def palette_metrics(a, b, k=6):
    """Whether the render reproduces the reference's dominant colors in similar proportions (perceptual color coverage)."""
    def key_colors(img):
        q = (img * 255 // 32).astype(int)
        flat = q.reshape(-1, 3)
        keys, counts = np.unique(flat, axis=0, return_counts=True)
        order = np.argsort(-counts)[:k]
        return keys[order] * 32 + 16, counts[order] / flat.shape[0]
    ka, pa = key_colors(a)
    kb, pb = key_colors(b)
    cover, prop_err = [], []
    for c, p in zip(ka, pa):
        d = np.sqrt(((kb - c) ** 2).sum(1))
        j = int(np.argmin(d))
        if d[j] < 60:
            cover.append(1.0)
            prop_err.append(abs(pb[j] - p))
        else:
            cover.append(0.0)
            prop_err.append(p)
    return float(np.mean(cover)), float(np.mean(prop_err))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reference'); ap.add_argument('render')
    ap.add_argument('--json'); ap.add_argument('--crops', type=int, default=8)
    args = ap.parse_args()
    ref0 = Image.open(args.reference).convert('RGB')
    a = load(args.reference)
    b = load(args.render, size=ref0.size)
    out = {}
    out['psnr'] = round(psnr(a, b), 3)
    try:
        from skimage.metrics import structural_similarity as ssim
        s = lambda x, y: ssim(x, y, channel_axis=2, data_range=1.0)
        out['ssim'] = round(s(a, b), 4)
        m, w = tile_scores(a, b, s)
        out['ssim_tile_mean'], out['ssim_tile_worst'] = round(m, 4), round(w, 4)
    except Exception:
        out['ssim'] = None
    try:  # Edge-structure F1 (Canny overlap, 3px tolerance).
        import cv2
        ea = cv2.Canny((a * 255).astype(np.uint8), 80, 160) > 0
        eb = cv2.Canny((b * 255).astype(np.uint8), 80, 160) > 0
        kern = np.ones((7, 7), np.uint8)
        da = cv2.dilate(ea.astype(np.uint8), kern) > 0
        db = cv2.dilate(eb.astype(np.uint8), kern) > 0
        prec = (eb & da).sum() / max(eb.sum(), 1)
        rec = (ea & db).sum() / max(ea.sum(), 1)
        out['edge_f1'] = round(2 * prec * rec / max(prec + rec, 1e-9), 4)
    except Exception:
        out['edge_f1'] = None
    try:  # Color-histogram EMD (mean per-channel 1D Wasserstein distance).
        from scipy.stats import wasserstein_distance
        d = [wasserstein_distance(a[..., c].ravel(), b[..., c].ravel()) for c in range(3)]
        out['color_emd'] = round(float(np.mean(d)), 5)
    except Exception:
        out['color_emd'] = None
    out['palette_coverage'], out['palette_prop_err'] = [round(v, 4) for v in palette_metrics(a, b)]
    try:  # Detail crops: SSIM/LPIPS for eight 2x zoom windows on a deterministic grid.
        from skimage.metrics import structural_similarity as ssim
        H, W = a.shape[:2]
        rng = [(i, j) for i in (0.15, 0.45, 0.7) for j in (0.15, 0.45, 0.7)][:args.crops]
        cs = []
        for fy, fx in rng:
            y, x = int(fy * H), int(fx * W)
            h, w = H // 5, W // 5
            ca, cb = a[y:y+h, x:x+w], b[y:y+h, x:x+w]
            if ca.size and cb.size:
                cs.append(ssim(ca, cb, channel_axis=2, data_range=1.0))
        out['detail_crop_ssim_mean'] = round(float(np.mean(cs)), 4)
        out['detail_crop_ssim_worst'] = round(float(np.min(cs)), 4)
    except Exception:
        out['detail_crop_ssim_mean'] = None
    try:
        import lpips, torch
        net = lpips.LPIPS(net='alex', verbose=False)
        ta = torch.tensor(a * 2 - 1).permute(2, 0, 1)[None].float()
        tb = torch.tensor(b * 2 - 1).permute(2, 0, 1)[None].float()
        out['lpips'] = round(float(net(ta, tb)), 4)
    except Exception:
        out['lpips'] = None
    try:
        import open_clip, torch
        model, _, pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        with torch.no_grad():
            fa = model.encode_image(pre(Image.fromarray((a*255).astype(np.uint8)))[None])
            fb = model.encode_image(pre(Image.fromarray((b*255).astype(np.uint8)))[None])
            out['clip_sim'] = round(float(torch.cosine_similarity(fa, fb)), 4)
    except Exception:
        out['clip_sim'] = None
    print(json.dumps(out, ensure_ascii=False))
    if args.json:
        json.dump(out, open(args.json, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
