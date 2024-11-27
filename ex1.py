import cv2
import matplotlib.image as mpimg

# 画像ファイルの読み込み
img = cv2.imread('ex1.jpg')

from ultralytics import YOLO

model = YOLO("yolov8x-pose.pt")

results = model(img, save=True, save_txt=True, save_conf=True)
keypoints = results[0].keypoints
print(keypoints.data[0][0])

skeleton=[[6,5],[6,8],[8,10],[5,7],[7,9],[6,12],[5,11],[12,11],[12,14],[14,16],[11,13],[13,15]]
for i in range(0,12):
    start_point=skeleton[i][0]
    end_point=skeleton[i][1]
    cv2.line(img, (int(keypoints.data[0][start_point][0]), int(keypoints.data[0][start_point][1])),(int(keypoints.data[0][end_point][0]),int(keypoints.data[0][end_point][1])), (0,0,255),thickness=4)

for i in range(5,17):
    cv2.circle(img, (int(keypoints.data[0][i][0]),int(keypoints.data[0][i][1])), 5, (0,255,255), thickness=-1)

# 画像を表示
cv2.imshow("ex1",img)
cv2.waitKey(0)