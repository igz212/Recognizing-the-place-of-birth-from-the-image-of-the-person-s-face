import cv2
import iGzFaceToolKit
'''
save images from whole videos in a directory. crop the and save them by movie name
to read whole images from directory
'''

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

iGzFaceToolKit.readListOfFilesAndWriteToTextFile("C:/source/Whole Movies/","./listfile.txt")
for file in iGzFaceToolKit.readFileList("listfile.txt"):
    iGzFaceToolKit.capture_cropFaces_save_FromMovie(file,"C:/Erfolgen2/")

iGzFaceToolKit.readListOfFilesAndWriteToTextFile("C:/source/Whole Images/","./listfile.txt")
for file in iGzFaceToolKit.readFileList("listfile.txt"):
    iGzFaceToolKit.cropFaces_save_FromImage("C:/source/whole Images/",file,"C:/Erfolgen3/")


