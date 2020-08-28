import cv2
import math
import argparse

def detectFace(data, space, configuration=0.7):
    frameOpencvDnn=space.copy()
    Height=frameOpencvDnn.shape[0]
    Width=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    data.setInput(blob)
    detections=data.forward()
    faceBoxes=[]
    for a in range(detections.shape[2]):
        confidence=detections[0,0,a,2]
        if confidence>configuration:
            x1=int(detections[0,0,a,3]*Width)
            y1=int(detections[0,0,a,4]*Height)
            x2=int(detections[0,0,a,5]*Width)
            y2=int(detections[0,0,a,6]*Height)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(Height/150)), 8)
    return frameOpencvDnn,faceBoxes

faceData="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageData="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderData="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceData)
ageNet=cv2.dnn.readNet(ageModel,ageData)
genderNet=cv2.dnn.readNet(genderModel,genderData)

video=cv2.VideoCapture(args.image if args.image else 0)
filler=20
while cv2.waitKey(1)<0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    Img,Boxes=detectFace(faceNet,frame)
    if not Boxes:
        print("No face detected")

    for Box in Boxes:
        face=frame[max(0,Box[1]-filler):
                   min(Box[3]+filler,frame.shape[0]-1),max(0,Box[0]-filler)
                   :min(Box[2]+filler, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(Img, f'{gender}, {age}', (Box[0], Box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", Img)
