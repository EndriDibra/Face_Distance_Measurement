# Author: Endri Dibra

# importing the required libraries
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

# getting camera of the device
camera = cv2.VideoCapture(0)

# creating an object for face detection operations
detector = FaceMeshDetector(maxFaces=1)

while True:

    # reading the output of camera
    success, img = camera.read()

    # getting face on the screen
    img, faces = detector.findFaceMesh(img, draw=True)

    # working only if face is detected
    if faces:

        # getting points from face
        face = faces[0]

        pointLeft = tuple(face[145])
        pointRight = tuple(face[374])

        # drawing one point for each eye and a line between them
        # to calculate distance
        #cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
        #cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        #cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)

        # getting the pixels distance
        pixels_distance, _ = detector.findDistance(pointLeft, pointRight)

        # finding the focal length
        W = 6.3
        #d = 50
        #f = (pixels_distance*d)/W
        #print(f)

        # finding the distance(depth)
        # average value of focal length
        f = 840
        depth = (W*f) / pixels_distance
        print(depth)

        # displaying output on the screen
        cvzone.putTextRect(img, f'Depth: {int(depth)}', (face[10][0]-135, face[10][1]-60), scale=3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)