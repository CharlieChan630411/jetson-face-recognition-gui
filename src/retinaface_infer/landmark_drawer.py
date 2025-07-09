import cv2
from typing import Sequence, List

def draw_landmarks(
    img,
    landmarks: Sequence[Sequence[float]],
    color: tuple = (0, 255, 0),
    radius: int = 2,
    thickness: int = -1,
):
    """Draw 5‑point facial landmarks on a BGR image.

    Parameters
    ----------
    img : numpy.ndarray
        Input/output BGR image. The function draws *in‑place* and also returns the image
        for convenience so you may chain calls.
    landmarks : Sequence[Sequence[float]]
        Iterable where each element is a length‑10 list/array representing the 5
        facial landmark points of one face in the order::

            [left_eye_x,  left_eye_y,
             right_eye_x, right_eye_y,
             nose_x,      nose_y,
             left_mouth_x,left_mouth_y,
             right_mouth_x,right_mouth_y]

        Coordinates are expected in image‑pixel space (i.e. after任何必要的縮放/偏移)。
    color : tuple(int, int, int), default (0, 255, 0)
        BGR 顏色。
    radius : int, default 2
        圓點半徑 (px)。
    thickness : int, default ‑1
        OpenCV thickness 參數；‐1 為填滿圓點。

    Returns
    -------
    numpy.ndarray
        同一張影像（已畫點）。
    """
    if img is None:
        raise ValueError("img 為 None，請檢查前處理流程是否正確讀取圖片。")

    for lm in landmarks:
        if len(lm) != 10:
            # 跳過不符合格式的 landmark
            continue
        for i in range(0, 10, 2):
            x = int(lm[i])
            y = int(lm[i + 1])
            cv2.circle(img, (x, y), radius, color, thickness)

    return img

__all__: List[str] = ["draw_landmarks"]

