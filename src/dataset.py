
import keras

from statistics import *

def generate_dataset():

    with open("../input/fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)

    num_of_instances = lines.size
    print("number of instances: ", num_of_instances)
    print("instance length: ", len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1, num_of_instances):
        try:
            emotion, img, usage = lines[i].split(",")
            val = img.split(" ")
            pixels = np.array(val, 'float32')
            emotion = keras.utils.to_categorical(emotion, 7)

            if 'Training' in usage:
                y_train.append(emotion)
                x_train.append(pixels)
            elif 'PublicTest' in usage:
                y_test.append(emotion)
                x_test.append(pixels)
        except:
            print("", end="")

    return y_train, y_test, x_train, x_test

