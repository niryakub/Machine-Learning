import cv2
import matplotlib.pyplot as plt

img = cv2.imread("C:/Users/nirya/Videos/zebra_800.png", cv2.IMREAD_COLOR)
cv2.imshow('image', img)

cv2.waitKey(0)

# It is for removing/deleting created GUI window from screen
# and memory
cv2.destroyAllWindows()