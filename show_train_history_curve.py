import pickle
from configuration import EPOCHS
from prepare_data import show_history_curve


def load_train_history_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as pickle_file:
        train_history_list = pickle.load(pickle_file)
    return train_history_list


def show_train_history_curve(train_history_list):
    show_history_curve(train_history_list[0], train_history_list[1],
                       train_history_list[2], train_history_list[3], EPOCHS, 'InceptionV4-epochs-100')


if __name__ == '__main__':
    train_history_list = load_train_history_from_pickle('./saved_history_data/'
                                                        'InceptionV4/InceptionV4-epochs-100')
    # print(len(train_history_list))
    # print(len(train_history_list[0]))
    # print(train_history_list)
    show_train_history_curve(train_history_list)