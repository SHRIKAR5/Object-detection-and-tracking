import cv2
import numpy as np
import time

weights=r'C:/.../darknet-master/yolov3.weights'
cfg=r'C:/.../darknet-master/cfg/yolov3.cfg'
net = cv2.dnn.readNet(weights, cfg)
#net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo
classes = []
with open(r"C:/.../darknet-master/data/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

layer_names = net.getLayerNames()

#outputlayers = [layer_names[200 - 1] for i in net.getUnconnectedOutLayers()]

#net.getUnconnectedOutLayers() are >> array([[200],[227],[254]])
#layer_names[199]= 'yolo_82'
#layer_names[227]= 'yolo_94'
#layer_names[254]= 'yolo_106'

outputlayers=[]
for i in net.getUnconnectedOutLayers():
    outputlayers.append(layer_names[i[0] - 1])
outputlayers


#frame= cv2.imread(r'C:\Users\SHRIKAR DESAI\Desktop\data_scientist\AI\srk.jpg') 
frame= cv2.imread(r'C:\...\suv.jpg') 
frame.shape
frame= cv2.resize(frame,(600,800)) 
height,width,channels = frame.shape
#detecting objects
blob = cv2.dnn.blobFromImage(frame,1/255,(320,320),(0,0,0),swapRB=True,crop=False) #reduce 416 to 320    
#blob = cv2.dnn.blobFromImage(image, scalefactor:(1/255)=0.00392, size, bgr(grey), swapRB, crop) 
    
net.setInput(blob)
outs = net.forward(outputlayers)
#len(outs)
#print(outs[2])
class_ids=[]
confidences=[]
boxes=[]
outs[0][0]
#outs=[[0.1,0.,....,0.],[0.2,0.,....,0.],[0.3,0.,....,0.]]   in out 3 list are there normally
#1st out i.e.[0.1,0.,....,0.] 
#len(detection)
#len(detection[5:])
#from 5th num in list there are confidances of class_id and as in line 'scores = detection[5:]'
#length of detection is 85 last 80 positions are of class_id in the order given in classes, so np.argmax gives index which has max value (and not the actual max value)

#type(confidence)

#detection=[x,y,width,height,unknown,[5-85]are class_id]
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            #object detected
            center_x= int(detection[0]*width)
            center_y= int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            
            #rectangle co-ordinaters
            x=int(center_x - w/2)
            #x=center_x
            #y=center_y
            y=int(center_y - h/2)
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

#cv2.dnn.NMSBoxes(boxes,confidences,score_threshold,IOU_threshold) 
#if 1 obj is detected multiple times so [IOU= (Area of overlap / area of union) of all boxes] and in o/p 1 box is shown   
indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

colors= np.random.uniform(0,255,size=(len(classes),3))
font = cv2.FONT_HERSHEY_PLAIN

for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence= confidences[i]
        color = colors[class_ids[i]]
        #m=round((x+w)/2)
        #n=round((y+h)/2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y-10),font,1,(0,0,255),2)
        #cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y-10),font,1,(0,0,255),2)
       #cv2.putText(frame,label+" "+str(round(confidence,number of points after decimal)),(x,y-10),font,1,(0,0,255),2)

cv2.imshow("Image",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


#l=np.array([11,22,33,44,55])
#m=np.argmax(l)
#m





