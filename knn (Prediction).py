import cv2
import numpy as np
import os

#knn
def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train , test ,k=5):
    dist = []
    for i in range(train.shape[0]):
        ix=train[i,:-1]
        iy=train[i,-1]

        d=distance(test,ix)
        dist.append([d,iy])

    dk = sorted(dist,key = lambda x : x[0] )[:k]
    labels = np.array(dk)[:,-1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

cap = cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('C:\\Users\\vivek gautam\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml')

skip = 0
fdata = []
datafolder = "D:\AI\opencv\datafolder"
label=[]

Name_id = 0
names ={}


Name_id = 0
for file in os.listdir(datafolder):
    if file.endswith('.npy'):
        name = file.split('.')[0] 
        names[Name_id] = name 
        data_item = np.load(os.path.join(datafolder, file),allow_pickle=True)
        fdata.append(data_item)
        target = Name_id * np.ones((data_item.shape[0],))
        Name_id += 1
        label.append(target)

face_dataset = np.concatenate(fdata,axis=0)
face_lebel  =  np.concatenate(label,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_lebel.shape)

train_set = np.concatenate((face_dataset,face_lebel),axis=1)
print(train_set.shape)


while True:
    ret,frame = cap.read()
    if ret == False :
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:
        x,y,w,h = face
        offset = 10
        face_sec = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_sec = cv2.resize(face_sec,(100,100))


        output = knn(train_set,face_sec.flatten())
        predict_name = names[int(output)]
        cv2.putText(frame, predict_name,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("Faces",frame)
    key = cv2.waitKey(1) & 0xFF
    if(key == ord('q')):
       break

cap.release()
cv2.destroyAllWindows()