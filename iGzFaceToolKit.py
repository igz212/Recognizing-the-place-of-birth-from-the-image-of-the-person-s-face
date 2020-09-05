import cv2
import os

#class iGzFace

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#to detect faces from an image<crop them and save them at last
def   cropFaces_save_FromImage(path, imageName, writeAddress):
    print(path+imageName)
    img = cv2.imread(path+imageName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5, minSize=(30, 30))
    faces_detected = format(len(faces)) + " faces detected!"
    # print(faces_detected) # to show info about amountof faces had been detected
    iCount = 0
    for (x, y, w, h) in faces: # Draw a rectangle around the faces and crop faces in image
        iCount += 1
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3) #to draw a recatngel around faces
        #viewImage(img,faces_detected)
        crop_img = img[y:y + h, x:x + w]
        #viewImage(crop_img, 'faces_detected_cropped')
        editedCount = str(iCount).zfill(2)
        pathToWrite = writeAddress + imageName + "frame%s.jpg" % editedCount
        print(pathToWrite)
        cv2.imwrite(pathToWrite, crop_img)


#a function to cpature, crop ans save images simultaneously from a movie
def capture_cropFaces_save_FromMovie(path, movieName, writeAddress):
    vidCap = cv2.VideoCapture(path+movieName)
    success, image = vidCap.read()
    count = 0
    if success:
        print("...successful processing on " + movieName + "...")
    while success:
        success, image = vidCap.read()
        if count % 25 == 0:
            icount = count // 25
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                #viewImage(image, 'detected face')
                # Draw a rectangle around the faces and crop faces in image
                crop_img = image[y:y + h, x:x + w]
                editedCount = str(icount).zfill(4)
                pathToWrite = writeAddress + movieName + "frame%s.jpg" % editedCount
                print(pathToWrite)
                cv2.imwrite(pathToWrite, crop_img)  # save frame as JPEG file
        # if cv2.waitKey(10) == 27:  # exit if Escape is hit
        #     break
        count += 1
    vidCap.release()

#to Read a list of files in a directory and write to file
def readListOfFilesAndWriteToTextFile(dirPath, writeAddress):
    places = []
    for file in os.listdir(dirPath):
        places.append(file)

    with open(writeAddress, 'w') as filehandle:
        for listItem in places:
            filehandle.write('%s\n' % listItem)

# to open file and read the content in a list
def readFileList(fileName):
    placesRead = []
    with open(fileName, 'r') as fileHandle:
        for line in fileHandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]
            # add item to the list
            placesRead.append(currentPlace)
    return placesRead


#a function to view image in a new window
def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#to recize image by scale
def resizeImage_byScale(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

#to recize image by Width and Height
def resizeImage_byWidthHeight(image, width , height):
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

# bin_n shows the Number of bins
def hog(img, bin_n = 16):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

#declare a function to get score from a model
def get_modelScore(model, xTrain, xTest, yTrain, yTest,printScore = False):
    model.fit(xTrain,yTrain)
    score = model.score(xTest, yTest)
    if (printScore == True):
        print(model.score(xTest, yTest))
    return score


def get_cross_val_score(model, x, y, k_to_fold = 3):
    folds = KFold(n_splits = k_to_fold)
    score_history = []
    # calculate k-fold cross validation scores in a loop
    for trainIndex, testIndex in folds.split(x):
        x_train, x_test, y_train, y_test = x[trainIndex], x[testIndex], y[trainIndex], y[testIndex]
        score_history.append(get_modelScore(model, x_train, x_test, y_train, y_test,printScore=True))
        print(score_history)
    return score_history


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        # If local neighbourhood pixel value is greater than or equal to center pixel values then set it to 1
        if img[x][y] >= center:
            new_value = 1
    except:
        # Exception is required when neighbourhood value of a center pixel value is null i.e. values present at boundaries.
        pass

    return new_value


def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    # top_left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))
    # top
    val_ar.append(get_pixel(img, center, x - 1, y))
    # top_right
    val_ar.append(get_pixel(img, center, x - 1, y + 1))
    # right
    val_ar.append(get_pixel(img, center, x, y + 1))
    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))
    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y - 1))
    # left
    val_ar.append(get_pixel(img, center, x, y - 1))
    # Now, we need to convert binary values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val

def lbp(imagePath):
    img_bgr = cv2.imread(imagePath, 1)
    height, width, _ = img_bgr.shape
    # We need to convert RGB image into gray one because gray image has one channel only.
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Create a numpy array as the same height and width of RGB image
    img_lbp = np.zeros((height, width), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    return img_lbp

