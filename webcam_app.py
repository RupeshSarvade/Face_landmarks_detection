import cv2
import mediapipe as mp
import time

vid = cv2.VideoCapture(0)
vid.set(3,640) #width id 3 and size 640
vid.set(4,480) #height id 4 and size 480
vid.set(10,100) #brightness id 10 and ratio 100
ptime=0 #previous time

#objects for facemesh
mpdraw = mp.solutions.drawing_utils
mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh(max_num_faces=10) #number of faces to be detected
drawspec = mpdraw.DrawingSpec(thickness=1,circle_radius=2,color=(0,255,255)) #(245,117,66)


while True:
    success,image = vid.read()
    imgrgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #facemesh requires rgb imgs/videos
    results = facemesh.process(imgrgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks: #display lanmarks on faces
            mpdraw.draw_landmarks(image,face_landmarks,mpfacemesh.FACEMESH_CONTOURS,drawspec,drawspec)

            for id,lm in enumerate(face_landmarks.landmark): #get landmarks positions on x,y,z
                # print(lm)
                ih,iw,ic = image.shape #image_height.width and channels
                x,y,z = int(lm.x*iw),int(lm.y*ih),int(lm.z*ic)
                print(f"id:{id},width:{x},height:{y},channels:{z}")

    ctime = time.time() #current time
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(image,f"FPS: {int(fps)}",(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),3) #for displaying fps on output screen
    cv2.imshow("image",image)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
