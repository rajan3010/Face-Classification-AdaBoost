import cv2
import math

DATA_FILE_PATH ='FDDB-folds'
IMAGE_PATH='images/'
def processandstore(data,length):
    face_list=[]
    nonface_list=[]
    for  imageannot in data:
        '''print(IMAGE_PATH+imageannot[0].rstrip()+'.jpg')
        exit()'''
        img=cv2.imread(IMAGE_PATH+imageannot[0].rstrip()+".jpg",-1)
        for faces in imageannot[1]:
            if len(face_list)>=length:
                return face_list, nonface_list
            face_attributes=faces.split()
            square_centre_x=int(math.floor(float(face_attributes[3])))
            square_centre_y=int(math.floor(float(face_attributes[4])))
            major_axis=int(math.floor(float(face_attributes[0])))
            if major_axis<75 or (square_centre_y-major_axis)<0 or (square_centre_x-major_axis)<0 or (square_centre_y+major_axis)>img.shape[0] or (square_centre_x+major_axis)>img.shape[1]:
                continue
            #print(major_axis)
            facesqr=img[square_centre_y-major_axis:square_centre_y+major_axis,square_centre_x-major_axis:square_centre_x+major_axis]
            nonfacesqr=img[0:100,0:100]
            if facesqr.ndim>2:
                facesqr=cv2.cvtColor(facesqr,cv2.COLOR_BGR2GRAY)
                nonfacesqr=cv2.cvtColor(nonfacesqr,cv2.COLOR_BGR2GRAY)
            facesqr=cv2.resize(facesqr,(20,20))
            nonfacesqr=cv2.resize(nonfacesqr,(20,20))
            '''cv2.imshow("nonfacesqr",nonfacesqr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
            '''if(len(facesqr.shape)<3):
                print("found")'''
            face_list.append(facesqr)
            nonface_list.append(nonfacesqr)

    return face_list, nonface_list

def getfacedata(start=1):
    face_data=[]
    for index in range(start,10):
        with open(DATA_FILE_PATH+"/FDDB-fold-0"+str(index)+"-ellipseList.txt") as f:
            lines=f.readlines()
            i=0
            while i<len(lines):
                if len(face_data)>=2300:
                    return index,face_data

                image=lines[i]
                i=i+1
                no_of_faces=int(lines[i])
                i=i+1
                ellipse_params=lines[i:i+no_of_faces]
                face_data.append((image,ellipse_params))
                i=i+no_of_faces

    return index,face_data

def loadandwrite():
    train_index,train_info=getfacedata()
    test_index, test_info=getfacedata(train_index+1)
    training_data_face, training_data_nonface=processandstore(train_info,500)
    testing_data_face, testing_data_nonface=processandstore(test_info,500)
    return training_data_face,training_data_nonface,testing_data_face,testing_data_nonface

#loadandwrite()