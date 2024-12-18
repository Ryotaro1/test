import cv2
from ultralytics import YOLO
import math

# 動画ファイルのパス
video_path = "ex3b.mp4"

# 動画を開く
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Failed to open video.")
    exit()

# YOLO モデルのロード
model = YOLO("yolov8x-pose.pt")

# スケルトンの接続リスト
skeleton = [[5, 6], [6, 8], [8, 10], [5, 7], [7, 9], [5, 11], [6, 12], [11, 12],
            [11, 13], [13, 15], [12, 14], [14, 16]]

# 角度を求める関数
def cos_formula(p1, p2, p3):
    a = math.dist(p1, p2)
    b = math.dist(p1, p3)
    c = math.dist(p2, p3)
    cos_c = (a**2 + b**2 - c**2) / (2 * a * b)
    acos_c = math.acos(cos_c)
    return math.degrees(acos_c)

# フレーム処理ループ
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # フレームでモデル推論を実行
    results = model(frame, save=False)
    keypoints = results[0].keypoints

    # キーポイントが存在する場合のみ処理
    if keypoints is not None:
        keypoints_data = keypoints.data[0]  # 最初の検出結果
        
        # Right-elbow (8), Right-shoulder (6), Right-hip (12) の角度を計算
        try:
            p1 = (keypoints_data[6][0], keypoints_data[6][1])  # Right-shoulder
            p2 = (keypoints_data[8][0], keypoints_data[8][1])  # Right-elbow
            p3 = (keypoints_data[12][0], keypoints_data[12][1])  # Right-hip
            
            angle = cos_formula(p1, p2, p3)
            
            # 条件に応じたスケルトン描画
            for start, end in skeleton:
                x1, y1 = int(keypoints_data[start][0]), int(keypoints_data[start][1])
                x2, y2 = int(keypoints_data[end][0]), int(keypoints_data[end][1])

                # 右腕の骨格 (肩-肘, 肘-手) を赤色で描画
                if 80 <= angle <= 100 and ((start == 6 and end == 8) or (start == 8 and end == 10)):
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=5)
                else:
                    # 他の骨格は青色で描画
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        except IndexError:
            # キーポイントが不足している場合はスキップ
            continue

    # フレームを表示
    cv2.imshow("Pose Detection", frame)

    # ESCキーで終了
    if cv2.waitKey(20) & 0xFF == 27:
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
