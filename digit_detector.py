import cv2
import numpy as np
from sklearn import svm, metrics

def get_digit_classifier(digits):
    # modified from http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
    # To apply a classifier on this data, we need to flatten the image, to turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    #print("+++++++++++++",digits.images[0],"+++++++++++++")
    data = digits.images.reshape((n_samples, -1))
    #print("************", data[0], "************")

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001, probability=True)

    # learn the digitss
    classifier.fit(data, digits.target)

    #print("Classification report for classifier %s:\n%s\n"
    #      % (classifier, metrics.classification_report(expected, predicted)))
    #print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    return classifier


def detect_digit(frame, classifier):
    # HIER NINAS TEIL
    # Eingabe: Zugeschnittenes Schwarz-Weißes OpenCV Bild "frame" mit genau einer oder keiner Ziffer

    if len(frame) < 8:
        return 0


    #Höhe/Breite bestimmen
    height, width = frame.shape[:2]


    #10% von w und h berechnen

    x=int(0.1*width)
    y=int(0.1*height)

    #frame-Zuschnitt

    frame_zuschnitt=frame[x:int(x+(width-2*x)) ,  y: int(y+(height-2*y)) ]

    # classifier needs an 8x8 picture with 16 greyscale steps (4 Bit) and black digits on white bg
    frame_mini = cv2.resize(frame_zuschnitt, (8,8))
    # now we can invert the image for black digits on white
    frame_mini = cv2.bitwise_not(frame_mini)
    # currently it has 256 Steps (8 Bit). Reduce amount of steps by dividing by (256/4) and convert to float64
    frame_mini = np.float64(frame_mini // 16)
    frame_mini = np.array(frame_mini, np.float64)

    # now the 8x8 picture has to be flatten:
    frame_mini_flat = np.ndarray.flatten(frame_mini)

    #print("~~~~~~~~~~~~~~",frame_mini_flat,"~~~~~~~~~~~~~~")
    predicted_probs = classifier.predict_proba([frame_mini_flat])[0]
    predicted_digit = np.argmax(predicted_probs)
    predicted_digit_prob = predicted_probs[predicted_digit]

    # print("Zahl:",predicted_digit,"%:",predicted_digit_prob)

    # Ausgabe: Integer der Ziffer oder 0 wenn (wahrscheinlich) keine Ziffer vorhanden
    return predicted_digit if predicted_digit_prob > 0.2 else 0
