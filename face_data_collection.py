import cv2
import numpy as np
print('Press Q. to switch of camera')
cap = cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('C:\\Users\\vivek gautam\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml')

skip=0
fdata = []
fsec=0

datafolder = "./datafolder"
data_path=input("Please Enter your name : ")
while True:
    ret,frame=cap.read()
    if ret == False:
        continue

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key=lambda f:f[2]*f[3])

    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        offset = 10
        fsec = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        fsec = cv2.resize(fsec,(100,100))
    
    skip+=1
    if(skip%10==0):
        fdata.append(fsec)
        print(len(fdata))
    cv2.imshow("Frames",frame)
    cv2.imshow("Face sec",fsec)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed ==ord('q'):
        break

fdata=np.asarray(fdata,dtype=object)
fdata=fdata.reshape((fdata.shape[0],-1))
print(fdata.shape)

np.save('./datafolder/'+data_path+'.npy',fdata)
print("Images saved successfully.....!!!!!")
print("Images is saved as : datafolder/",data_path)


cap.release()
cap.destroyAllWindows()
print('Quited...')

