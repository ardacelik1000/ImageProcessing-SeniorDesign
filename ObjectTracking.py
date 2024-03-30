import cv2
import numpy as np

# '0' activates the camera
camera = cv2.VideoCapture(0)

def main():

    while True:
        ret, image = camera.read()

        # Converting all color from bgr to hsv (hue, saturation, value)
        ThreshFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = (9,139,196,255,255,255)

        #This line creates a mask containing pixels that match the specified color range. 
        #This mask encodes pixels representing the tennis ball as white (255) and other pixels as black (0).
        ThreshForBall = cv2.inRange(ThreshFrame, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))
        kernel = np.ones((5,5),np.uint8)

        #to reduce noise and separate objects. This helps to remove white noise and small objects.
        mask = cv2.morphologyEx(ThreshForBall, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if  25> radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(image, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                cv2.putText(image,"("+str(center[0])+","+str(center[1])+")", (center[0]+10,center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)

        # show the frame to our screen
        cv2.imshow("Original", image)
        #cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == '__main__':
    main()