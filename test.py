import cv2
import numpy as np

#create a black image
#img = np.zeros((512, 512, 3), np.uint8) 
image = np.zeros((512, 512, 3), dtype=np.uint8)

#Draw a rectangle
cv2.rectangle(image, (100, 100), (400, 400), (255, 0, 0), -1)
#Draw a circle
cv2.circle(image, (256, 256), 100, (0, 255, 0), -1)

#Display the image
cv2.imshow('Image', image)

#Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
