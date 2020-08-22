import iGzFaceToolKit
import pandas as pd


#At the begining of partB I decide to resize whole croped face to 64x64 to have an standard size ready to compare by features
iGzFaceToolKit.readListOfFilesAndWriteToTextFile("C:/Erfolgen2/","./erfolgen2list.txt")
for file in iGzFaceToolKit.readFileList("erfolgen2list.txt"):
    img = cv2.imread("./Erfolgen2/"+file)
    cv2.imwrite("./Erfolgen2Resized64x64/"+file, iGzFaceToolKit.resizeImage_byWidthHeight(img,64,64))

iGzFaceToolKit.readListOfFilesAndWriteToTextFile("C:/Erfolgen3/","./erfolgen3list.txt")
for file in iGzFaceToolKit.readFileList("erfolgen3list.txt"):
    img = cv2.imread("./Erfolgen3/"+file)
    cv2.imwrite("./Erfolgen3Resized64x64/"+file, iGzFaceToolKit.resizeImage_byWidthHeight(img,64,64))


#to have a dictionary to map ID and label
mapID = iGzFaceToolKit.readFileList("mapID")
map_BP_Label = iGzFaceToolKit.readFileList("map_BP_Label")

fullMap_ID_to_Label ={}
for i in range (342):
    tempDictionary = {mapID[i]:int(map_BP_Label[i])}
    fullMap_ID_to_Label.update(tempDictionary)

#print(fullMap_ID_to_Label)


erfolgen2 = iGzFaceToolKit.readFileList("./erfolgen2list.txt")
labelSet2 = iGzFaceToolKit.readFileList("./erfolgen2label.txt")
iCount = 0

#make prefered dataSet of HOG results and origin class label
with open("./dataSet2.txt", 'w') as filehandle:
    for file in erfolgen2:
        img = cv2.imread("./erfolgen2/" + file)
        hog_result = iGzFaceToolKit.hog(img)
        for hogNum in hog_result:
            filehandle.write('%s,' % hogNum)
        filehandle.write('%s\n' % labelSet2[iCount])
        iCount += 1

erfolgen3 = iGzFaceToolKit.readFileList("./erfolgen3list.txt")
labelSet3 = iGzFaceToolKit.readFileList("./erfolgen3label.txt")
iCount = 0

with open("./dataSet3.txt", 'w') as filehandle:
    for file in erfolgen3:
        img = cv2.imread("./erfolgen3/" + file)
        hog_result = iGzFaceToolKit.hog(img)
        for hogNum in hog_result:
            filehandle.write('%s,' % hogNum)
        filehandle.write('%s\n' % labelSet3[iCount])
        #print("[",iCount,"]:",labelSet3[iCount])
        iCount += 1

#note : at the begining sould add a line based on tuple name to have a suitabe text file to read by pandas 
data =  pd.read_csv('dataSet2Pandas.txt')
data_np_array = np.asarray(data)

#thess dimensions are choosen upon dataset dimensions
X = data_np_array[:, :63]
y = data_np_array[:,64]
#to be inforemed
#print(y)
#print(type(X))
#print(type(y))
#print(X.shape)
#print(y.shape)

#to split the dataset in train and test-set for the following training and prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 15)


#different kernel functions to make model
linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')
linear.fit(X_train, y_train)
rbf = svm.SVC(kernel='rbf', gamma=1, C=1 , decision_function_shape='ovo')
rbf.fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo')
poly.fit(X_train, y_train)
sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo')
sig.fit(X_train, y_train)



#make predictions on the test data set using our 4 different kernel functions:
linear_pred = linear.predict(X_test)
rbf_pred = rbf.predict(X_test)
poly_pred = poly.predict(X_test)
sig_pred = sig.predict(X_test)

# utilize a performance measure — accuracy, To understand how well they perform
#                retrieve the accuracy and print it for all 4 kernel functions:
accuracy_lin = linear.score(X_test, y_test)
accuracy_rbf = rbf.score(X_test, y_test)
accuracy_poly = poly.score(X_test, y_test)
accuracy_sig = sig.score(X_test, y_test)

print("Accuracy Linear Kernel:", accuracy_lin)
print("Accuracy Radial Basis Kernel:", accuracy_rbf)
print("Accuracy Polynomial Kernel:", accuracy_poly)
print("Accuracy Sigmoid Kernel:", accuracy_sig)
