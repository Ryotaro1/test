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

max_area=0
max_box=None
for box in boxes:
    x1,y1,x2,y2=(box.data[0][0],box.data[0][1],box.data[0][2],box.data[0][3])
    area=(x2-x1)*(y2-y1)

    if area > max_area:
        max_area = area
        max_box = (x1,y1,x2,y2)

if max_box:
    x1, y1, x2, y2 = map(int, max_box)
    img_2 = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255))

# 画像を表示
cv2.imwrite("ex2_output.jpg",img_2)
cv2.imshow("ex2",img)
cv2.waitKey(0)