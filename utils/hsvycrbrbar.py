

import cv2
import numpy as np

def nothing(x):
    pass

# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
# For HSV
cv2.createTrackbar('H Lower','image',0,179,nothing)
cv2.createTrackbar('H Upper','image',179,179,nothing)
cv2.createTrackbar('S Lower','image',0,255,nothing)
cv2.createTrackbar('S Upper','image',255,255,nothing)
cv2.createTrackbar('V Lower','image',0,255,nothing)
cv2.createTrackbar('V Upper','image',255,255,nothing)

# For YCrCb
cv2.createTrackbar('Y Lower','image',0,255,nothing)
cv2.createTrackbar('Y Upper','image',255,255,nothing)
cv2.createTrackbar('Cr Lower','image',0,255,nothing)
cv2.createTrackbar('Cr Upper','image',255,255,nothing)
cv2.createTrackbar('Cb Lower','image',0,255,nothing)
cv2.createTrackbar('Cb Upper','image',255,255,nothing)

# Load your image
image = cv2.imread('1.jpg')

while(1):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Get current positions of trackbars for HSV
    hL = cv2.getTrackbarPos('H Lower','image')
    hU = cv2.getTrackbarPos('H Upper','image')
    sL = cv2.getTrackbarPos('S Lower','image')
    sU = cv2.getTrackbarPos('S Upper','image')
    vL = cv2.getTrackbarPos('V Lower','image')
    vU = cv2.getTrackbarPos('V Upper','image')

    # Get current positions of trackbars for YCrCb
    yL = cv2.getTrackbarPos('Y Lower','image')
    yU = cv2.getTrackbarPos('Y Upper','image')
    crL = cv2.getTrackbarPos('Cr Lower','image')
    crU = cv2.getTrackbarPos('Cr Upper','image')
    cbL = cv2.getTrackbarPos('Cb Lower','image')
    cbU = cv2.getTrackbarPos('Cb Upper','image')

    # Set the lower and upper HSV range according to the value selected
    lower_hsv = np.array([hL, sL, vL])
    upper_hsv = np.array([hU, sU, vU])

    # Set the lower and upper YCrCb range according to the value selected
    lower_ycrcb = np.array([yL, crL, cbL])
    upper_ycrcb = np.array([yU, crU, cbU])

    # Apply the thresholds
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)

    # Bitwise-AND mask and original image
    result_hsv = cv2.bitwise_and(image, image, mask=mask_hsv)
    result_ycrcb = cv2.bitwise_and(image, image, mask=mask_ycrcb)
    combined_result = cv2.bitwise_and(image, image, mask=combined_mask)

    # Show the images
    cv2.imshow('Original', image)
    # cv2.imshow('HSV Mask', mask_hsv)
    # cv2.imshow('YCrCb Mask', mask_ycrcb)
    # cv2.imshow('Combined Mask', combined_mask)
    # cv2.imshow('HSV Result', result_hsv)
    # cv2.imshow('YCrCb Result', result_ycrcb)
    cv2.imshow('Combined Result', combined_result)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
