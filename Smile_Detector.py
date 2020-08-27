import cv2
face_detector=cv2.CascadeClassifier('C:/Users/KIIT/Desktop/Smile Detection/haarcascade_frontalface_default.xml')
smile_detector=cv2.CascadeClassifier('C:/Users/KIIT/Desktop/Smile Detection/haarcascade_smile.xml')
webcam=cv2.VideoCapture(0)

while True:
    successful_frame_read, frame=webcam.read()
    if not successful_frame_read:
        break

    #Change to grayscale
    frame_grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #convert frame to grayscale
    #Detect multi scale
    faces=face_detector.detectMultiScale(frame_grayscale)
    #print(faces) [[554 322 664 222]]
    for (x,y,w,h) in faces:
        #Draw a rectangle around the face
        cv2.rectangle(frame,(x,y), (x+w,y+h), (100,200,50), 3)
        the_face=frame[y:y+h,x:x+w] # assinging the_face with that individual face values x.y.w.h respecitvely and individually


        #Change to grayscale
        face_grayscale=cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY) #convert frame to grayscale



        smiles=smile_detector.detectMultiScale(face_grayscale,scaleFactor=1.7,minNeighbors=20)

        # for (x_,y_,w_,h_) in smiles:           
        #     cv2.rectangle(the_face,(x_,y_), (x_+w_,y_+h_), (100,20,220), 3)
        if len(smiles)>0:
            cv2.putText(frame,'smiling',(x, y+h+40), fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))
        


    cv2.imshow('Smile Detector', frame) #pass grayscale instead of frame
    cv2.waitKey(1) # this will give a single frame if you domt give any milisecond in argument
webcam.release()
cv2.destroyAllWindows()    