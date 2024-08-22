import csv
import datetime
import time

import keras as keras
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import confusion_matrix
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


class StopOnPoint(tf.keras.callbacks.Callback):
    """This stops the model when the validation accuracy is above a point"""
    def __init__(self, point):
        super(StopOnPoint, self).__init__()
        self.point = point

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs["val_acc"]
        if accuracy >= self.point:
            self.model.stop_training = True


def average_model(height):
    """
    This builds the model
    :param height: The height of the FSM
    """
    i = keras.layers.Input(shape=(height, 4, 3), name='input_layer')
    max1 = layers.AveragePooling2D((1, 4))(i)
    max1out = layers.Conv2D(3, 3, padding='same', activation='ReLU', kernel_regularizer=keras.regularizers.l2(0.0001))(
        max1)
    max1out = tf.image.resize(max1out, (height, 4))
    # max1out = layers.Conv2D(3, 3, padding='same', activation='ReLU', kernel_regularizer=keras.regularizers.l2(0.0001))(max1out)
    max2 = layers.AveragePooling2D((1, 2))(i)

    max2out = layers.Conv2D(3, 3, padding='same', activation='ReLU', kernel_regularizer=keras.regularizers.l2(0.0001))(
        max2)
    max2out = tf.image.resize(max2out, (height, 4))
    # max2out = layers.Conv2D(3, 3, padding='same', activation='ReLU', kernel_regularizer=keras.regularizers.l2(0.001))(max2out)

    max3out = layers.Conv2D(3, (4, 3), padding='same', activation='ReLU',
                            kernel_regularizer=keras.regularizers.l2(0.0001))(i)
    max3out = layers.Conv2D(3, (4, 3), padding='same', activation='ReLU',
                            kernel_regularizer=keras.regularizers.l2(0.0001))(max3out)

    max4 = layers.AveragePooling2D((height, 1))(i)
    max4out = layers.Conv2D(3, (4, 3), padding='same', activation='ReLU',
                            kernel_regularizer=keras.regularizers.l2(0.0001))(max4)
    # max4out = layers.Conv2D(3, 3, padding='same', activation='ReLU', kernel_regularizer=keras.regularizers.l2(0.001))(max4out)
    max4out = tf.image.resize(max4out, (height, 4))

    con = layers.concatenate([max1out, max2out, max3out, max4out], axis=2)
    con = layers.Conv2D(3, 3, padding='same', activation='ReLU', kernel_regularizer=keras.regularizers.l2(0.0001))(con)

    # out = layers.AveragePooling2D((10,4))(con)
    out = layers.Flatten()(con)
    # out =  tf.keras.layers.Dense(100, activation= 'ReLU')(out)
    # out =  tf.keras.layers.Dense(50, activation= 'ReLU')(out)
    out = tf.keras.layers.Dense(2, activation='sigmoid')(out)

    model = tf.keras.Model(inputs=i, outputs=out)
    return model


def scheduler(epoch, lr):
    """
    This function reduces the learning rate over time
    :param epoch: The current epoch number
    :param lr: The current learning rate
    :return: This new learning rate
    """
    tf.summary.scalar('learning rate', data=lr, step=epoch)
    with open("lr.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow([lr])
    if epoch < 6:
        return lr
    else:
        return lr * tf.math.exp(-0.03)


class LimitTrainingTime(tf.keras.callbacks.Callback):
    """This sets a limit on the amount of time the model can train for"""
    def __init__(self, max_time_s):
        super().__init__()
        self.max_time_s = max_time_s
        self.start_time = None

    def on_train_begin(self, logs):
        self.start_time = time.time()

    def on_train_batch_end(self, batch, logs):
        now = time.time()
        if now - self.start_time > self.max_time_s:
            self.model.stop_training = True


def train_model(lr, height, model_file):
    """
    This creates and trains the model
    :param lr: learning rate
    :param height: height of the FSM
    :param model_file: model file name
    """
    batch_size = 64

    data_root = "Dataset"

    # height *= 5
    image_shape = (height, 4)

    training_data_dir = str(data_root)
    datagen_kwargs = dict(rescale=1. / 255, validation_split=.40, fill_mode='nearest')

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(  # This creates the validation dataset
        training_data_dir,
        subset="validation",
        shuffle=True,
        target_size=image_shape,
        batch_size=batch_size,

    )
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    train_generator = train_datagen.flow_from_directory(  # This creates the training dataset
        training_data_dir,
        subset="training",
        shuffle=True,
        target_size=image_shape,
        batch_size=batch_size)

    image_batch, label_batch = [], []
    for image_batch, label_batch in train_generator:
        break

    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=((batch_size, height, 4, 3), (batch_size, 2))).prefetch(tf.data.AUTOTUNE)
    valid_dataset = tf.data.Dataset.from_generator(
        lambda: valid_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=((batch_size, height, 4, 3), (batch_size, 2))).prefetch(tf.data.AUTOTUNE)

    print(image_batch.shape, label_batch.shape)

    print(train_generator.class_indices)

    num_epochs = 100000

    # lr = 0.1
    train_length = train_generator.samples
    steps_per_epoch = train_length // batch_size

    val_sub_splits = 2
    test_length = valid_generator.samples
    validation_steps = test_length // batch_size // val_sub_splits

    model = average_model(height)
    # model = covModel(height)
    early_stop_ac = [StopOnPoint(0.95)]
    model.summary()
    checkpoint_path = f'models/{model_file}.ckpt'

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1,
                                                     monitor='val_loss', mode='min')
    learning_rate = lr
    # learning_rate = 6e-05
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.BinaryAccuracy(
                      name='binary_accuracy', dtype=None, threshold=0.5
                  )])
    call_back_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    print(model.output.shape)
    print(label_batch[0].shape)
    log_dir = str("Model Training History") + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='500,520')
    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    callback_time = LimitTrainingTime(1200)
    # model.evaluate(train_generator,batch_size=batch_size, steps=steps_per_epoch)
    model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
              validation_data=valid_dataset,
              callbacks=[cp_callback, tensorboard_callback, call_back_lr, early_stop_ac, callback_time], shuffle=True)  #

    pred = model.predict(train_generator)
    i = 0

    numpy_labels = np.empty([0, 2])

    for images, labels in train_generator:  # only take first element of dataset

        numpy_labels = np.append(numpy_labels, labels, axis=0)
        if i >= len(train_generator) - 1:
            break
        i += 1

    con = confusion_matrix(y_true=numpy_labels.argmax(axis=1), y_pred=pred.argmax(axis=1))
    del model
    print(con)
    return con
