modifiedFrame = cv2.GaussianBlur(modifiedFrame, (3, 3), 0)
        hsv = cv2.cvtColor(modifiedFrame, cv2.COLOR_BGR2HSV)  # convert it to hsv

        h, s, v = cv2.split(hsv)
        v += 255
        final_hsv = cv2.merge((h, s, v))

        modifiedFrame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def sortFunc(c):
    x, y = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)[0][0]
    #split image in 9 rows and cols and determine which the field is most likely in
    h, w, c = frame.shape
    gridsize = 9
    rowh = int(h/gridsize)
    colw = int(w/gridsize)
    for i in range(1,gridsize):
        # Spalte bestimmen
        if (i - 1) * colw <= x+int(colw/2) < i * colw:
            col = i
        # Reihe bestimmen
        if (i-1)*rowh <= y+int(rowh/2) < i*rowh:
            row = i
    cv2.circle(thresh_draw, (x+int(colw/2), y+int(rowh/2)), 1, (255,255,255))
    print(col, row, "=", col+row*1000)
    return col + row*1000


#cnts = sorted(cnts, key=sortFunc)


    """
        Detect Squares

    # black color boundaries (R,B,G)
    lower = np.array([0, 0, 0], dtype="uint8")
    upper = np.array([120, 120, 120], dtype="uint8")

    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(modifiedFrame, lower, upper)
    output = cv2.bitwise_and(modifiedFrame, modifiedFrame, mask=mask)

    ret, thresh = cv2.threshold(mask, 40, 255, 0)


    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        # cv2.drawContours(output, contours, -1, 255, 3)

        # find the biggest area
        c = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(c)
        # draw the book contour (in green)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop_img = output[y:y + h, x:x + w]
    """

    # ocr
    filename = os.path.abspath("{}.png".format(os.getpid()))
    print(filename)
    cv2.imwrite(filename, modifiedFrame)
    # add tesseract to path?
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    print("detected:", text)