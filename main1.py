import cv2
img=cv2.imread("C:/Users/User/Downloads/03.jpg");
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,th=cv2.threshold(gray,150,255,cv2.THRESH_TOZERO_INV)
cv2.imshow('image',th)
cv2.waitKey(0)
cv2.destroyAllWindows()

#plt.imshow(img2)
#plt.axis('off')
#plt.show()
#cv2.imshow('org',img)
#cv2.waitKey(0)
#venv\\Scripts\\activate.bat
""""
from google.colab.patches import cv2_imshow
for i in range(6):
plt.subplot(2,3,i+1),olt.imshow(img[i],'gray',vmin=0,vmax=255)
plt.title(title[i])
plt.xticks([]),plt.ticks([])
plt.show()
for img in images
cv2."""


