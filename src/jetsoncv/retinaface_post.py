import numpy as np

def _prior_box(w: int, h: int) -> np.ndarray:
    priors = []
    for stride in [8, 16, 32]:
        fm_w, fm_h = int(np.ceil(w / stride)), int(np.ceil(h / stride))
        for j in range(fm_h):
            for i in range(fm_w):
                cx = (i + 0.5) * stride / w
                cy = (j + 0.5) * stride / h
                s = stride / w
                priors.append([cx, cy, s, s])
    return np.array(priors, dtype=np.float32)

_PRIORS = _prior_box(640, 608)
_VARIANCE = np.array([0.1, 0.2], dtype=np.float32)

def decode(boxes, priors):
    cxcy = priors[:, :2] + boxes[:, :2] * _VARIANCE[0] * priors[:, 2:]
    wh   = priors[:, 2:] * np.exp(boxes[:, 2:] * _VARIANCE[1])
    tl   = cxcy - wh / 2
    br   = cxcy + wh / 2
    return np.hstack([tl, br])

def decode_landm(landms, priors):
    xy = priors[:, :2]; wh = priors[:, 2:]
    out = np.zeros_like(landms)
    for i in range(5):
        out[:, 2*i:2*i+2] = xy + landms[:, 2*i:2*i+2] * _VARIANCE[0] * wh
    return out

def nms(dets, scores, thresh=0.4):
    x1,y1,x2,y2 = dets.T
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size>0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1); h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        iou = inter/(areas[i]+areas[order[1:]]-inter)
        order = order[np.where(iou<=thresh)[0]+1]
    return keep
