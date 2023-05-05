import math
import cv2 as cv

image = cv.imread("assets/shape.png")

width = 800
ratio = width / image.shape[1]
height = int(image.shape[0] * ratio)

image = cv.resize(image, (width, height))
new_image = cv.resize(image, (width, height))

new_image = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)

_, threshold = cv.threshold(new_image, 127, 255, cv.THRESH_BINARY)

contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

first = False
for contour in contours:
    if first == False:
        first = True
        continue

    approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
    M = cv.moments(contour)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    if len(approx) == 3:
        cv.putText(image, "Triangle", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
    elif len(approx) == 4:
        cv.putText(
            image,
            "Rectangle OR Square",
            (x, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
        )
    elif len(approx) == 5:
        cv.putText(image, "Pentagon", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
    elif len(approx) == 6:
        cv.putText(image, "Hexagon", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
    elif len(approx) == 7:
        cv.putText(image, "Heptagon", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
    elif len(approx) == 8:
        cv.putText(image, "Octagon", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
    elif len(approx) == 9:
        cv.putText(image, "Nonagon", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
    elif len(approx) == 10:
        cv.putText(image, "Decagon", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
    else:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity > 0.9:
            cv.putText(image, "Circle", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))

cv.imshow("Shape Detection", image)
cv.waitKey()
