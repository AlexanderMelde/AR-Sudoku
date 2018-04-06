import cv2
import numpy as np
from sklearn import svm, metrics

def get_digit_classifier(digits):
    images_and_labels = list(zip(digits.images, digits.target))
    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    #print("+++++++++++++",digits.images[0],"+++++++++++++")
    data = digits.images.reshape((n_samples, -1))
    #print("************", data[0], "************")

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)

    # We learn the digits on the first half of the digits
    classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

    # Now predict the value of the digit on the second half:
    expected = digits.target[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    return classifier


def detect_digit(frame, classifier):
    # HIER NINAS TEIL
    # Eingabe: Zugeschnittenes Schwarz-Weißes OpenCV Bild "frame" mit genau einer oder keiner Ziffer
    #
    if len(frame) < 8:
        return 0

    # classifier needs an 8x8 picture with 16 greyscale steps (4 Bit) and black digits on white bg
    frame_mini = cv2.resize(frame, (8,8))
    # now we can invert the image for black digits on white
    frame_mini = cv2.bitwise_not(frame_mini)
    # currently it has 256 Steps (8 Bit). Reduce amount of steps by dividing by (256/4) and convert to float64
    frame_mini = np.float64(frame_mini // 16)
    frame_mini = np.array(frame_mini, np.float64)

    #Remove border pixels
    frame_mini[0:1, : ] = 0
    frame_mini[ : ,0:1] = 0

    # now the 8x8 picture has to be flatten:
    frame_mini_flat = np.ndarray.flatten(frame_mini)

    #print("~~~~~~~~~~~~~~",frame_mini_flat,"~~~~~~~~~~~~~~")
    predicted = classifier.predict([frame_mini_flat])

    #print("Zahl:",predicted)

    """
    # (works, but slow and no good detection (not all numbers)
    filename = os.path.abspath(str("{}_"+str(row-1)+str(col-1)+".png").format(os.getpid()))
    cv2.imwrite(filename, frame)
    # Bei Fehler muss installiert werden: "AR-Sudoku/tesseract-ocr-setup...exe" und der Installationsordner muss zum System-Path hinzugefügt werden
    text = pytesseract.image_to_string(Image.open(filename), config='outputbase digits')
    # os.remove(filename)
    print(filename, "detected:", text)
    if text == "":
        return 0
    return text
    """
    # Ausgabe: Integer der Ziffer oder 0 wenn keine Ziffer vorhanden
    return predicted[0]
