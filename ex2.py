import cv2

# 画像ファイルの読み込み
img = cv2.imread('ex2.jpg')

from ultralytics import YOLO

model = YOLO("yolov8x.pt")

results = model(img, save=True, save_txt=True, save_conf=True)
boxes = results[0].boxes
for box in boxes:
    print("box",box.data)
print("name",results[0].names)
print(boxes[0].data[0][0])

for i in range(len(boxes)):
    img_2 = cv2.rectangle(img,(int(boxes[i].data[0][0]),int(boxes[i].data[0][1])),(int(boxes[i].data[0][2]),int(boxes[i].data[0][3])),(0,0,255))

# 画像を表示
cv2.imwrite("ex2_output.jpg",img_2)
cv2.imshow("ex2",img)
cv2.waitKey(0)