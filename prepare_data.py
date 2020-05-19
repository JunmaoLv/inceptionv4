import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import math
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    BATCH_SIZE, train_tfrecord, valid_tfrecord, test_tfrecord, \
    model_name_list, model_index, EPOCHS, cross_validate_tfrecord, k_fold
from parse_tfrecord import get_parsed_dataset


def load_and_preprocess_image(image_raw, data_augmentation=False):
    # decode
    image_tensor = tf.io.decode_image(contents=image_raw, channels=CHANNELS, dtype=tf.dtypes.float32)
    image_tensor = tf.image.per_image_standardization(image_tensor)
    if data_augmentation:
        image = tf.image.random_flip_left_right(image=image_tensor)
        image = tf.image.resize_with_crop_or_pad(image=image,
                                                 target_height=int(IMAGE_HEIGHT * 1.2),
                                                 target_width=int(IMAGE_WIDTH * 1.2))
        image = tf.image.random_crop(value=image, size=[IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])
        image = tf.image.random_brightness(image=image, max_delta=0.5)
    else:
        image = tf.image.resize(image_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])

    return image


def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return all_image_path, all_image_label


def get_the_length_of_dataset(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count


def generate_datasets():
    train_dataset = get_parsed_dataset(tfrecord_name=train_tfrecord)
    valid_dataset = get_parsed_dataset(tfrecord_name=valid_tfrecord)
    test_dataset = get_parsed_dataset(tfrecord_name=test_tfrecord)

    train_count = get_the_length_of_dataset(train_dataset)
    valid_count = get_the_length_of_dataset(valid_dataset)
    test_count = get_the_length_of_dataset(test_dataset)

    # read the dataset in the form of batch
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)

    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count


def generate_cross_validate_dataset():
    cross_validate_dataset = get_parsed_dataset(tfrecord_name=cross_validate_tfrecord)
    test_dataset = get_parsed_dataset(tfrecord_name=test_tfrecord)

    cross_validate_dataset_count = get_the_length_of_dataset(cross_validate_dataset)
    test_count = get_the_length_of_dataset(test_dataset)

    cross_validate_dataset = cross_validate_dataset.batch(batch_size=BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)
    batch_num = math.ceil(cross_validate_dataset_count / BATCH_SIZE)
    boundary = math.ceil(batch_num / k_fold)
    cross_validate_dataset_list = []
    for i in range(k_fold):
        cross_validate_dataset_item = []
        for index, cross_validate_item in enumerate(cross_validate_dataset):
            if i * boundary <= index < (i+1) * boundary:
                cross_validate_dataset_item.append(cross_validate_item)
            else:
                continue
        # print('batch个数: {}'.format(len(cross_validate_dataset_item)))
        cross_validate_dataset_list.append(cross_validate_dataset_item)
    # print('交叉验证集合的个数: {}, 每个验证集合的batch个数: {}, 每个batch的类型: {}'
    #       .format(len(cross_validate_dataset_list), len(cross_validate_dataset_list[0]),
    #               type(cross_validate_dataset_list[0][0])))

    return cross_validate_dataset_list, test_dataset, cross_validate_dataset_count, test_count


def show_history_curve(train_loss_array, train_accuracy_array, valid_loss_array, valid_accuracy_array,
                       epochs, network_title):
    plt.figure()
    x = np.linspace(1, epochs, num=epochs, endpoint=True)
    # print(x)
    plt.plot(x, train_loss_array, color='red', label='train loss')
    plt.plot(x, train_accuracy_array, color='red', linestyle='--', label='train acc')
    plt.plot(x, valid_loss_array, color='blue', label='valid loss')
    plt.plot(x, valid_accuracy_array, color='blue', linestyle='--', label='valid acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy and loss')
    plt.title(network_title)
    plt.legend()
    plt.savefig('saved_train_history_picture/{}.png'.format(network_title), dpi=500)
    plt.show()


def test_tf_batch_num(train_dataset, batch_size):
    train_dataset_batch = train_dataset.batch(batch_size=batch_size)
    for batch_size_num, train_sample in enumerate(train_dataset_batch):
        index = 0
        image_raw_list = train_sample['image_raw'].numpy()
        for _ in image_raw_list:
            index = index + 1
        print('第{}个batch : {}个样本'.format(batch_size_num, index))


def test_image_standard():
    train_dataset = get_parsed_dataset('dataset/train.tfrecord')
    train_dataset_batch = train_dataset.batch(batch_size=4)
    index = 0
    for train_sample in train_dataset_batch:
        image_tensor = []
        image_raw_list = train_sample['image_raw'].numpy()
        image_label = train_sample['label'].numpy()
        for image_raw_item in image_raw_list:
            # image_data = tf.io.decode_image(contents=image_raw_item, channels=3, dtype=tf.dtypes.float32)
            image_data = load_and_preprocess_image(image_raw_item, data_augmentation=True)
            # plt.figure()
            # plt.imshow(image_data)
            # plt.show()
            image_tensor.append(image_data)
            print('{} : max:{}, min:{}, mean:{}, , shape:{}'.format(index, np.max(image_data), np.min(image_data),
                                                                    np.mean(image_data), image_data.shape))
            # print('shape:{}'.format(image_data.shape))
            index = index + 1
        images = tf.stack(image_tensor, axis=0)
        print('{} : {}'.format(image_label, images.shape))


if __name__ == '__main__':
    # images_path, images_label = get_images_and_labels('./dataset/train/')
    # print('images path : {} images label : {}'.format(images_path, images_label))
    generate_cross_validate_dataset()

