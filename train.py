from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
import pickle
import numpy as np
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    EPOCHS, BATCH_SIZE, save_model_dir, model_index, model_name_list, k_fold
from prepare_data import generate_datasets, load_and_preprocess_image, \
    show_history_curve, generate_cross_validate_dataset
import math
from models import inception_v4


def get_model():
    return inception_v4.InceptionV4()


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


def process_features(features, data_augmentation):
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()

    return images, labels


def set_gpu_growth():
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
    # gpu setting
    set_gpu_growth()

    # create model
    model = get_model()
    # print the structure of the model
    print_model_summary(network=model)

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # loss_object = tf.keras.losses.BinaryCrossentropy()
    # optimizer = tf.keras.optimizers.RMSprop()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    # train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    # define the history array
    train_loss_array = []
    train_accuracy_array = []
    valid_loss_array = []
    valid_accuracy_array = []

    # define saved flag
    cross_validate_flag = 'cross-validate'

    # @tf.function
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=True)
            loss = loss_object(y_true=label_batch, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # @tf.function
    def valid_step(image_batch, label_batch):
        predictions = model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)


    def start_training(cross_validate=False):
        if cross_validate:
            cross_validate_dataset_list, test_dataset, cross_validate_dataset_count, test_count = \
                generate_cross_validate_dataset()
            train_count = cross_validate_dataset_count / k_fold * (k_fold - 1)
            for epoch in range(EPOCHS):
                train_loss_k_fold = []
                train_accuracy_k_fold = []
                valid_loss_k_fold = []
                valid_accuracy_k_fold = []
                for k in range(k_fold):
                    step = 0
                    train_dataset_k_fold = []
                    valid_dataset_k_fold = cross_validate_dataset_list[k]
                    for index, cross_validate_dataset_item in enumerate(cross_validate_dataset_list):
                        if index != k:
                            train_dataset_k_fold = train_dataset_k_fold + cross_validate_dataset_item

                    for train_batch in train_dataset_k_fold:
                        step += 1
                        # images, labels = process_features(features, data_augmentation=True)
                        images, labels = process_features(train_batch, data_augmentation=False)
                        train_step(images, labels)
                        print("cross validation: {}/{}, Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".
                              format(k+1, k_fold, epoch + 1, EPOCHS, step, math.ceil(train_count / BATCH_SIZE),
                                     train_loss.result().numpy(), train_accuracy.result().numpy()))

                    for train_batch in valid_dataset_k_fold:
                        valid_images, valid_labels = process_features(train_batch, data_augmentation=False)
                        valid_step(valid_images, valid_labels)

                    print("cross validation: {}/{}, Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                          "valid loss: {:.5f}, valid accuracy: {:.5f}".format(k+1, k_fold, epoch + 1,
                                                                              EPOCHS,
                                                                              train_loss.result().numpy(),
                                                                              train_accuracy.result().numpy(),
                                                                              valid_loss.result().numpy(),
                                                                              valid_accuracy.result().numpy()))

                    train_loss_k_fold.append(train_loss.result().numpy())
                    train_accuracy_k_fold.append(train_accuracy.result().numpy())
                    valid_loss_k_fold.append(valid_loss.result().numpy())
                    valid_accuracy_k_fold.append(valid_accuracy.result().numpy())

                    train_loss.reset_states()
                    train_accuracy.reset_states()
                    valid_loss.reset_states()
                    valid_accuracy.reset_states()

                print("after cross validation: Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                      "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                          EPOCHS,
                                                                          np.mean(train_loss_k_fold),
                                                                          np.mean(train_accuracy_k_fold),
                                                                          np.mean(valid_loss_k_fold),
                                                                          np.mean(valid_accuracy_k_fold)))

                train_loss_array.append(np.mean(train_loss_k_fold))
                train_accuracy_array.append(np.mean(train_accuracy_k_fold))
                valid_loss_array.append(np.mean(valid_loss_k_fold))
                valid_accuracy_array.append(np.mean(valid_accuracy_k_fold))
        else:
            global cross_validate_flag
            cross_validate_flag = 'no-cross-validate'
            train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
            for epoch in range(EPOCHS):
                step = 0
                for train_batch in train_dataset:
                    step += 1
                    # images, labels = process_features(features, data_augmentation=True)
                    images, labels = process_features(train_batch, data_augmentation=False)
                    train_step(images, labels)
                    print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(
                        epoch + 1, EPOCHS, step, math.ceil(train_count / BATCH_SIZE),
                        train_loss.result().numpy(), train_accuracy.result().numpy()))

                for train_batch in valid_dataset:
                    valid_images, valid_labels = process_features(train_batch, data_augmentation=False)
                    valid_step(valid_images, valid_labels)

                print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                      "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                          EPOCHS,
                                                                          train_loss.result().numpy(),
                                                                          train_accuracy.result().numpy(),
                                                                          valid_loss.result().numpy(),
                                                                          valid_accuracy.result().numpy()))
                train_loss_array.append(train_loss.result().numpy())
                train_accuracy_array.append(train_accuracy.result().numpy())
                valid_loss_array.append(valid_loss.result().numpy())
                valid_accuracy_array.append(valid_accuracy.result().numpy())

                train_loss.reset_states()
                train_accuracy.reset_states()
                valid_loss.reset_states()
                valid_accuracy.reset_states()


    # start_training(cross_validate=False)
    start_training(cross_validate=True)
    # if epoch % save_every_n_epoch == 0:
    # model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')

    # save weights
    filepath = save_model_dir + '{}-epoch-{}-batch-{}-{}/'.format(model_name_list[model_index],
                                                                  EPOCHS, BATCH_SIZE, cross_validate_flag)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    model.save_weights(filepath=filepath+'{}-epochs-{}-batch-{}'.format(model_name_list[model_index],
                                                                        EPOCHS, BATCH_SIZE),
                       save_format='tf')

    # save the train history to pickle file
    pickle_path = 'saved_history_data/{}-{}/'.format(model_name_list[model_index], cross_validate_flag)
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)
    history_array = [train_loss_array, train_accuracy_array, valid_loss_array, valid_accuracy_array]
    with open(pickle_path + '{}-epochs-{}-batch-{}'.format(model_name_list[model_index], EPOCHS,
                                                           BATCH_SIZE), 'wb') as pickle_file:
        pickle.dump(history_array, pickle_file)

    # show the train curve
    # show_history_curve(train_loss_array, train_accuracy_array,
    #                    valid_loss_array, valid_accuracy_array,
    #                    EPOCHS, 'train history of {}-epochs-{}'.format(model_name_list[model_index], EPOCHS))

    # save the whole model
    # tf.saved_model.save(model, save_model_dir)

    # convert to tensorflow lite format
    # model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)

