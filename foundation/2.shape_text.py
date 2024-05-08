import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)
#print(img)


######################## DRAWING ON IMAGE ############################

cv2.line(img,(0,0),(250,500),(0,255,0),3)       #draw line      
#cv2.line( {image}, {beginning_point}, {ending_point}, {color}, {thickness}, {linetype} )

cv2.rectangle(img,(0,0),(250,350),(0,0,255),2)  #draw rectangle 
#cv2.rectangle( {image}, {beginning_point}, {ending_point}, {color}, {thickness}, {linetype} )


cv2.circle(img,(400,50),30,(255,255,0),5)       #draw circle    
#cv2.circle( {image}, {centre}, {radius}, {color}, {thickness}, {linetype} )




######################## WRITING ON IMAGE ############################
cv2.putText(img," OPENCV  ",(300,200),cv2.FONT_HERSHEY_COMPLEX,2,(0,150,0),3)
#cv2.putText( {img}, {text}, {beginning_point}, {font_style}, {font_size},  {color}, {thickness} )


cv2.imshow("Image",img)
cv2.waitKey(0)