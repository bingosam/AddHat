# -*- encoding=utf8 -*-
__author__ = "Zhang kunbin"

from add_hat import *


img_name = "test"
img = cv2.imread("data/images/" + img_name + ".jpg")
hatter = Hatter(u"data/images/hat.png", mini_head_degrees=2, debug=True)
output, face_count = hatter.add_hat(img)

cv2.imshow("output", output)
cv2.waitKey(0)
cv2.imwrite("data/result/" + img_name + ".jpg", output)
cv2.destroyAllWindows()
