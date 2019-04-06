
import cv2
vc=cv2.VideoCapture("Rails.mp4")
c=1
if vc.isOpened():
    rval,frame=vc.read()
else:
    rval=False
while rval:
    rval,frame=vc.read()
    if c < 10 :
        cv2.imwrite('./pic/'+str(c)+'.jpg',frame)
    c=c+1
    if c>10:
        break
    cv2.waitKey(1)
vc.release()
