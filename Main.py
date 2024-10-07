import cv2
import numpy as np
import CT

path = r"C:\Users\Admin\demo\Grading\test2.JPG"
widthImg = 700
heightImg = 800
CorrectAns = [[1, 3, 3, 2, 2, 3, 2, 3, 4, 4, 1, 3, 3, 3, 2, 3, 1, 4, 1, 4, 2, 1, 3, 2, 3],
                [2, 3, 1, 3, 2, 4, 3, 4, 4, 4, 3, 2, 3, 4, 1, 2, 1, 1, 1, 3, 4, 2, 2, 3, 1],
                [4, 2, 4, 3, 3, 3, 2, 2, 1, 1, 2, 1, 1, 2, 3, 4, 4, 4, 4, 3, 3, 2, 3, 4, 3],
                [1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 3, 2, 3, 4, 1, 2, 3, 2, 1, 2, 3, 2, 1, 3, 3]]

#IMPORT IMAGE
img = cv2.imread(path)
img = cv2.resize(img, (widthImg, heightImg))

imgContour = img.copy()
imgAnswer = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlurr = cv2.GaussianBlur(imgGray, (5,5), 1)
imgCanny = cv2.Canny(imgBlurr, 10, 100)

#FIND ALL CONTOUR
contours, hierachy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContour, contours, -1, (0,0,255), 2)

rectContour = CT.rectContour(contours)

#LOOP OVER 4 SHAPES ANSWER
totalGrading = []
CorrectArrayImg = []
for shape in range(0,4):
    
    answer = CT.getCornerPoints(rectContour[shape])

    # print(answer1.size)
    if answer.size != 0:
        cv2.drawContours(imgAnswer, answer, -1, (0,255,0), 10)

        Answer = CT.reorder(answer)

        # print(Answer)
        pt1 = np.float32(Answer)
        # print(pt1)
        pt2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)
        ImgWarp = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        ImgWarpGray = cv2.cvtColor(ImgWarp, cv2.COLOR_BGR2GRAY)
        ImgWarpThresh = cv2.threshold(ImgWarpGray, 150, 255, cv2.THRESH_BINARY_INV)[1]
        
        #SPLIT BOXES
        boxes = CT.splitBoxes(ImgWarpThresh)
        # cv2.imshow("test", boxes[3])
        # cv2.waitKey(0)

        myPixelVal = np.zeros((25,5))
        countC = 0
        countR = 0

        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countR][countC] = totalPixels
            countC += 1
            if (countC == 5): countR += 1; countC = 0
        # print(myPixelVal)

        #FINDING THE INDEX VALUES OF THE MARKINGS
        myIndex = []
        for i in range(0, 25):
            arr = myPixelVal[i]
            myIndexVal = np.where(arr==np.amax(arr))
            myIndex.append(myIndexVal[0][0])
        # print(myIndex)

        #GRADING THE POINTS
        grading = []
        for i in range(0,25):
            if CorrectAns[shape][i] == myIndex[i]:
                grading.append(1)
            else:
                grading.append(0)
        ImgResult = ImgWarp.copy()
        ImgResult = CT.ShowAnswers(ImgResult, myIndex, grading, CorrectAns[shape], 25, 5)
        CorrectArrayImg.append(ImgResult)

        # print(grading)
    totalGrading.append(sum(grading))
print(sum(totalGrading))

imgStacked = CT.stackImages(CorrectArrayImg, 0.5)

cv2.imshow("Test", imgStacked)
cv2.waitKey(0)

cv2.imwrite('TestResult.jpg', imgStacked)
