import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto.
    return tf.io.parse_single_example(example_proto, {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    })


def get_parsed_dataset(tfrecord_name):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    parsed_dataset = raw_dataset.map(_parse_image_function)
    return parsed_dataset


if __name__ == '__main__':
    train_dataset = get_parsed_dataset('dataset/valid.tfrecord')
    train_dataset_batch = train_dataset.batch(batch_size=5)
    index = 0
    for train_sample in train_dataset_batch:
        image_tensor = []
        image_raw_list = train_sample['image_raw'].numpy()
        image_label = train_sample['label'].numpy()
        for image_raw_item in image_raw_list:
            image_data = tf.io.decode_image(contents=image_raw_item, channels=3, dtype=tf.dtypes.float32)
            # plt.figure()
            # plt.imshow(image_data)
            # plt.show()
            image_data = tf.image.per_image_standardization(image_data)
            image_tensor.append(image_data)
            print('{} : max:{}, min:{}, mean:{}'.format(index, np.max(image_data), np.min(image_data), np.mean(image_data)))
            index = index + 1
        images = tf.stack(image_tensor, axis=0)
        print('{} : {}'.format(image_label, images.shape))
