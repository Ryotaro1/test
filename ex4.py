import cv2
import numpy as np
from ultralytics import YOLO
import math

# ファイルパス
template_path = "ex1.jpg"
video_path = "ex3b.mp4"

# YOLO モデルのロード
model = YOLO("yolov8x-pose.pt")

# 骨格間の差分を計算する関数
def calculate_skeleton_difference(kp1, kp2):
    # 両方のキー点の数が一致している前提で処理
    total_diff = 0
    for point1, point2 in zip(kp1, kp2):
        if point1[2] > 0.5 and point2[2] > 0.5:  # 信頼度スコアが十分高い点のみを比較
            diff = math.dist(point1[:2], point2[:2])  # ユークリッド距離
            total_diff += diff
    return total_diff

# テンプレート画像の読み込みと骨格推定
template_image = cv2.imread(template_path)
if template_image is None:
    print("Failed to load template image.")
    exit()

template_results = model(template_image)
if len(template_results) == 0 or template_results[0].keypoints is None:
    print("Failed to extract keypoints from template.")
    exit()

template_keypoints = template_results[0].keypoints.data[0]  # 最初の検出結果

# 動画の読み込み
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Failed to open video.")
    exit()

frame_number = 0
min_difference = float('inf')
best_frame = -1

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_number += 1

    # 動画フレームでの骨格推定
    results = model(frame)
    if len(results) == 0 or results[0].keypoints is None:
        continue  # 骨格が検出されない場合はスキップ

    keypoints = results[0].keypoints.data[0]  # 最初の検出結果

    # 差分を計算
    difference = calculate_skeleton_difference(template_keypoints, keypoints)

    # 差分が最小なら記録
    if difference < min_difference:
        min_difference = difference
        best_frame = frame_number

    # 進捗を表示
    print(f"Frame {frame_number}: Difference = {difference:.2f}")

cap.release()

# 結果の出力
if best_frame != -1:
    print(f"The frame with the smallest difference is: {best_frame} (Difference = {min_difference:.2f})")
else:
    print("No matching frames found.")
