from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, MaxPool2D, BatchNormalization, \
    Activation
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras


def generate_first_model(num_classes, batch_size, epochs, x_train, y_train):
    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))
    # ------------------------------
    # batch process
    gen = ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

    # ------------------------------

    model.compile(loss='categorical_crossentropy'
                  , optimizer=keras.optimizers.Adam()
                  , metrics=['accuracy']
                  )

    # model.fit_generator(x_train, y_train, epochs=epochs) #train for all trainset
    model.fit_generator(train_generator, steps_per_epoch=batch_size,
                        epochs=epochs)  # train for randomly selected one

    return model


def generate_second_model(num_classes, batch_size, epochs, x_train, y_train):
    model = Sequential()
    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), input_shape=(48, 48, 1), activation='softmax', padding='same'))
    model.add(MaxPooling2D(pool_size=(5, 5)))

    # 2nd convolution layer
    model.add(Conv2D(128, (5, 5), activation='softmax', padding='same'))
    model.add(Conv2D(128, (5, 5), activation='softmax', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024, activation='softmax'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='softmax'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))
    # ------------------------------
    # batch process
    gen = ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

    model.compile(loss='categorical_crossentropy'
                  , optimizer=keras.optimizers.Adadelta()
                  , metrics=['accuracy']
                  )
    model.fit_generator(train_generator, steps_per_epoch=batch_size,
                        epochs=epochs)
    return model