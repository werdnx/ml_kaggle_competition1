import cv2
import numpy as np
from scipy import signal

#=========================================================================
# Locate all components
#=========================================================================
def locateComponents(img):
    """Extracts all components from an image"""

    out = img.copy()
    res = cv2.findContours(np.uint8(out.copy()),\
                 cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0][0]

    ret = []
    row, col = out.shape
    minSiz = 8

    for cnt in contours:
        # get bounding box
        y, x, n, m = cv2.boundingRect(cnt)
        # check area
        #if m < minSiz or n < minSiz:
         #   continue
        #end if

        ret.append(np.int32([x, x+m, y, y+n]))
        out = cv2.rectangle(out, (y,x), (y+n,x+m), (255,255,255), 2)

    #end for

    return ret, out

# end function

#=========================================================================
# TESTING
#=========================================================================

img = cv2.imread('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/ISIC_0052212.jpg', 0)

regions, out = locateComponents(img)
cv2.imwrite('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/ISIC_0052212_1.jpg', out)
print regions

cv2.imshow('Given image', img)
cv2.imshow('Located regions', out)
cv2.waitKey(0)