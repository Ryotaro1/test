import cv2
import numpy as np
from ultralytics import YOLO

# YOLOモデルのロード
model = YOLO("yolov8x.pt")

# 画像の読み込み
img = cv2.imread("ex2.jpg")

# YOLOで物体検出
results = model(img)
boxes = results[0].boxes

# 青色ジャージの検出
def is_blue_area(img, box):
    """
    検出ボックス内の色が青系かを判定する関数
    """
    # ボックスの座標取得
    x1, y1, x2, y2 = map(int, box.data[0][:4])
    cropped_img = img[y1:y2, x1:x2]  # ボックス領域の切り抜き

    # HSV色空間に変換
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    # 青色の範囲を定義（HSVで）
    lower_blue = np.array([100, 50, 100])  # 青色の下限
    upper_blue = np.array([140, 255, 255])  # 青色の上限

    # 青色ピクセルの割合を計算
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 青色が一定割合以上であればTrue
    return cv2.countNonZero(mask) > 0

# 青色ジャージの領域を赤枠で描画
for box in boxes:
    if is_blue_area(img, box):  # 青色領域かを判定
        x1, y1, x2, y2 = map(int, box.data[0][:4])
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 結果を保存および表示
cv2.imwrite("ex2_output.jpg", img)
cv2.imshow("Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
