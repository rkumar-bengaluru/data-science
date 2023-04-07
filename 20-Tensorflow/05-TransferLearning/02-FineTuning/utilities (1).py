import os
import matplotlib.pyplot as plt
import shutil
import zipfile
from urllib import request

def walk_dir(root_dir):
    """
    Prints no of images in each subdirectory starting from root_dir
    :param root_dir:
    :return:
    """
    for dir_path, dir_name, file_name in os.walk(root_dir):
        print(f'there are {len(dir_name)} directories and {len(file_name)} images in path {dir_path}')


def plot_loss_and_accuracy(history):
    """
    plot the loss curve and accuracy curve against each epochs.

    :param history:
    :return:
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    epochs = len(history.history['loss'])
    axes[0].plot(epochs, history.history['loss'], label='training loss')
    axes[0].plot(epochs, history.history['val_loss'], label='validation loss')
    axes[0].set_title('Loss Per Epochs')
    axes[0].set(xlabel='epochs', ylabel='loss')

    axes[1].plot(epochs, history.history['accuracy'], label='training accuracy')
    axes[1].plot(epochs, history.history['val_accuracy'], label='validation accuracy')
    axes[1].set_title('Accuracy Per Epochs')
    axes[1].set(xlabel='epochs', ylabel='Accuracy')

    plt.legend()

def download_file(file_name):
    file_to_download = file_name.split('/')[-1]
    if os.path.exists(file_to_download) == False:
        print(f'downloading file {file_to_download}')
        request.urlretrieve(file_name,file_to_download)
    else:
        print(f'file already exists {file_to_download}')


def unzip_file(file_name):
    file_to_unzip = file_name.split('/')[-1]
    zip_ref = zipfile.ZipFile(file_to_unzip)
    zip_ref.extractall()
    zip_ref.close()


def clean_up(file_name):
    file = file_name.split('/')[-1]
    dir_name = file.split('.')[0]
    # remove file
    if os.path.exists(file):
        print(f'removing file and directory {file}')
        os.remove(file)
    else:
        print(f'file does not exist {file}')
    # remove dir
    if os.path.exists(dir_name):
        print(f'directory exists {dir_name}')
        shutil.rmtree(dir_name)