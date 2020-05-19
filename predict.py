import tensorflow as tf
from configuration import save_model_dir, test_image_dir, model_index, EPOCHS, model_name_list
from prepare_data import load_and_preprocess_image
from train import get_model
import os
from timeit import default_timer as timer


def get_single_picture_prediction(model, picture_dir):
    image_tensor = load_and_preprocess_image(tf.io.read_file(filename=picture_dir), data_augmentation=False)
    image = tf.expand_dims(image_tensor, axis=0)
    prediction = model(image, training=False)
    pred_class = tf.math.argmax(prediction, axis=-1)
    return pred_class
    # return prediction


def get_list_picture_prediction(model, picture_dir):
    class_name_list = ['defect', 'no_defect']
    if 'no_defect' in picture_dir:
        picture_path_list = os.listdir(picture_dir)
    elif 'defect' in picture_dir:
        picture_path_list = os.listdir(picture_dir)
    image_tensor_list = []
    for picture_path_item in picture_path_list:
        picture_path = os.path.join(picture_dir, picture_path_item)
        image_tensor = load_and_preprocess_image(tf.io.read_file(filename=picture_path), data_augmentation=False)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    prediction = model(images, training=False)
    pred_class_num_list = tf.math.argmax(prediction, axis=-1)
    pred_class_name_list = []
    for pred_class_num_item in pred_class_num_list:
        pred_class_name_item = class_name_list[pred_class_num_item]
        pred_class_name_list.append(pred_class_name_item)
    return pred_class_name_list


if __name__ == '__main__':
    # GPU settings
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)

    # load the model
    model_create_start_time = timer()
    model = get_model()
    # model.load_weights(filepath=save_model_dir+"model")
    # model.load_weights(filepath=save_model_dir+"model_index-"+'{}-epochs-{}'.format(model_index, EPOCHS))
    model.load_weights(filepath=save_model_dir+'{}/{}-epochs-{}'.format(model_name_list[model_index],
                                                                        model_name_list[model_index], EPOCHS))
    model_create_end_time = timer()


    print('model create spend : {} seconds'.format(model_create_end_time - model_create_start_time))

    # pred_class = get_single_picture_prediction(model, test_image_dir)
    model_predict_start_time = timer()
    pred_class_name = get_list_picture_prediction(model, test_image_dir + 'no_defect/')
    model_predict_end_time = timer()
    # print('every predict spend : {} seconds'.format((model_predict_end_time - model_predict_start_time)
    #                                                 / len(pred_class_name)))
    print('5 predict spend : {} seconds'.format(model_predict_end_time - model_predict_start_time))
    print(pred_class_name)
