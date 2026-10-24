import numpy as np

def project(K, X):
    x = K @ X
    return x[:2] / x[2]

def test_crop_homography_consistency():
    # Arbitrary intrinsics and 3D points in camera coordinates.
    K = np.array([[800., 0, 700.], [0, 800., 480.], [0, 0, 1.]])
    pts = np.array([[0.3, -0.2, 2.0], [-1.1, 0.4, 3.5], [0.8, 0.9, 5.0]]).T
    x0, y0, s = 250., 120., 2.0
    H = np.array([[s, 0, -s * x0], [0, s, -s * y0], [0, 0, 1.]])
    Kc = H @ K
    for i in range(pts.shape[1]):
        full = project(K, pts[:, i])
        crop = project(Kc, pts[:, i])
        expect = (full - np.array([x0, y0])) * s
        assert np.allclose(crop, expect, atol=1e-9), (crop, expect)

def test_crop_roundtrip_scale():
    K = np.array([[600., 0, 512.], [0, 600., 384.], [0, 0, 1.]])
    x0, y0, s = 100., 50., 3.0
    H = np.array([[s, 0, -s * x0], [0, s, -s * y0], [0, 0, 1.]])
    Kc = H @ K
    # Angular distance per pixel in the child frustum = 1/s of that in the full frame.
    assert np.isclose(Kc[0, 0] / K[0, 0], s)
    assert np.isclose(Kc[1, 1] / K[1, 1], s)
