from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from dataset import *
from model import *
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from configuration import *
from datetime import datetime

if __name__ == "__main__":

    num_classes, batch_size, epochs = configuration()

    y_train, y_test, x_train, x_test = generate_dataset()

    # ------------------------------
    # data transformation for train and test sets
    x_train = np.array(x_train, 'float32')
    y_train = np.array(y_train, 'float32')
    x_test = np.array(x_test, 'float32')
    y_test = np.array(y_test, 'float32')

    x_train /= 255  # normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    x_test = x_test.astype('float32')

    start_time = datetime.now()
    model = generate_second_model(num_classes, batch_size, epochs, x_train, y_train)

    # overall evaluation
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', 100 * score[1])

    time_elapsed = datetime.now() - start_time
    print('Time elapsed for learning (hh:mm:ss.ms) {}'.format(time_elapsed))

    predictions = model.predict(x_test)

    pred_list = []
    actual_list = []

    for i in predictions:
        pred_list.append(np.argmax(i))

    for i in y_test:
        actual_list.append(np.argmax(i))

    cm = confusion_matrix(actual_list, pred_list)
    print(cm)
    category = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
    df_cm = pd.DataFrame(cm, index=[i for i in category],
                         columns=[i for i in category])
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})
    plt.show()

    TP = np.diag(np.array(cm))
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = []
    for i in range(num_classes):
        temp = np.delete(cm, i, 0)
        temp = np.delete(temp, i, 1)
        TN.append(sum(sum(temp)))

    TN = np.array(TN)

    TPR = TP / (TP + FN)
    TNR = TN / (FP + TN)
    ACC = (TP + TN) / (TP +FN + FP + TN)
    F1 = 2*TP / (2*TP +FP + FN)

    print("------------------TPR---------------------")
    print(TPR)
    print("------------------TNR---------------------")
    print(TNR)
    print("------------------ACC----------------------")
    print(ACC)
    print("-------------------F1----------------------")
    print(F1)

    def emotion_analysis(emotions):
        objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        y_pos = np.arange(len(objects))

        plt.bar(y_pos, emotions, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('percentage')
        plt.title('emotion')

        plt.show()


    predictions = model.predict(x_test)

    index = 0
    for i in predictions:
        if index < 30 and index >= 20:
            testing_img = np.array(x_test[index], 'float32')
            testing_img = testing_img.reshape([48, 48])

            plt.gray()
            plt.imshow(testing_img)
            plt.show()

            print(i)

            emotion_analysis(i)
            print("----------------------------------------------")
        index = index + 1

    start_time = datetime.now()

    img = image.load_img("../input/fear.png", color_mode = 'grayscale', target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x /= 255

    custom = model.predict(x)
    emotion_analysis(custom[0])

    x = np.array(x, 'float32')
    x = x.reshape([48, 48])

    plt.gray()
    plt.imshow(x)
    plt.show()

    time_elapsed = datetime.now() - start_time
    print('Time elapsed for testing (hh:mm:ss.ms) {}'.format(time_elapsed))

