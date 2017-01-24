import cv2
import numpy as np

f=np.load('./facedata.npy')
l=np.load('./facelabels.npy')

f=f.reshape((f.shape[0],f.shape[1]*f.shape[2]))
rgb = cv2.VideoCapture(0)             #capture frames 
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #xml to detect face
# print facec

font = cv2.FONT_HERSHEY_SIMPLEX

def recognize_face(im):            #convert image detected by cam to 100*100 gray 
    im = cv2.resize(im, (100, 100))
    im=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    #im = cv2.resize(im, (100, 100))
    im = im.flatten()              #convert image to to linear data
    
    def dist(x1, x2):
        return np.sqrt(((x1 - x2)**2).sum())


    def knn(X_train, x, y_train, k=7):

        vals = []
    
        for ix in range(X_train.shape[0]):
            v = [dist(x, X_train[ix, :]), y_train[ix]]
            vals.append(v)
    
        updated_vals = sorted(vals, key=lambda x: x[0])
        pred_arr = np.asarray(updated_vals[:k])
        pred_arr = np.unique(pred_arr[:, 1], return_counts=True)
        pred = pred_arr[1].argmax()
        # return pred_arr[0][pred]
        return pred_arr[0][pred]


    res = knn(f, im, l, 7)
    return res

while True:
    _, fr = rgb.read()
    
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    # print gray.shape
    faces = facec.detectMultiScale(gray, 1.3, 5)    #detect face 1.3,5 is sensitivity for cascasde file
    
    for (x,y,w,h) in faces:
        fc = fr[x:x+w, y:y+h, :]
        out = recognize_face(fc)         #calling knn
        cv2.putText(fr, out, (x, y), font, 1, (255, 255, 0), 2)  #write name
    	cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)    #create rectangle around face
    
    cv2.imshow('rgb', fr)    #window for rgb cam
    cv2.imshow('gray', gray) #window for gray cam
    if cv2.waitKey(1) == 27: #break loop when esc(27) key is pressed check every 1 sec
        break

cv2.destroyAllWindows()
