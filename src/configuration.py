import tensorflow as tf

def configuration():

    # cpu - gpu configuration
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 0, 'CPU': 56})  # max: 1 gpu, 56 cpu
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    # variables
    num_classes = 7  # angry, disgust, fear, happy, sad, surprise, neutral
    batch_size = 64
    epochs = 1

    return num_classes, batch_size, epochs