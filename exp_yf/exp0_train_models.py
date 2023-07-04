import sys
sys.path.append("/media/data/alexlemo/probabilistic_kt")
# pip install --force-reinstall torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = '1'

from tensorflow import keras
from torchsummary import summary

import torch.nn as nn
import torch.optim as optim
from nn.retrieval_evaluation import evaluate_model_retrieval
from nn.nn_utils import train_model, save_model, load_model

from models.yt import YT_Small, FaceNet
from exp_yf.yt_dataset import get_yt_loaders
from tqdm import tqdm

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def train_yf_model(net, learning_rates=[0.001, 0.0001], iters=[50, 50], output_path='facenet_yf.model'):
    """
    Trains a baseline (classification model)
    :param net: the network to be trained
    :param learning_rates: the learning rates to be used during the training
    :param iters: number of epochs using each of the supplied learning rates
    :param output_path: path to save the trained model
    :return:
    """


    # Load data
    train_loader, test_loader, database_loader = get_yt_loaders(batch_size=128, feature_type='transfer')

    # Define loss
    criterion = nn.CrossEntropyLoss()

    for lr, iter in zip(learning_rates, iters):
        print("Training with lr=%f for %d iters" % (lr, iter))
        optimizer = optim.Adam(net.parameters(), lr=lr)
        train_model(net, optimizer, criterion, train_loader, epochs=iter)
        save_model(net, output_file=output_path)


def train_yf_models():
    """
    Trains the baselines teacher/students
    :return:
    """

    # YT_Small training
    net = YT_Small(num_classes=533) 
    net.cuda()
    train_yf_model(net, learning_rates=[0.001, 0.0001], iters=[50, 50],
                        output_path='models/small_yf.model')

    # FaceNet training
    net = FaceNet(num_classes=533)
    net.cuda()
    train_yf_model(net, learning_rates=[0.001, 0.0001], iters=[50, 50],
                        output_path='models/facenet_yf.model')

def evaluate_yf_models_retrieval():
    """
    Evaluates the baselines teacher/students
    :return:
    """
    evaluate_model_retrieval(net=YT_Small(533), path='models/small_yf.model',
                             result_path='results/small_yf_baseline.pickle', dataset_name='yf', dataset_loader=get_yt_loaders)

    evaluate_model_retrieval(net=FaceNet(533), path='models/facenet_yf.model',
                   result_path='results/facenet_yf_baseline.pickle', dataset_name='yf', dataset_loader=get_yt_loaders)

def print_model_summary(model, path):
    model.cuda()
    load_model(model, path)
    summary(model, input_size=(3,64,64))


if __name__ == '__main__':
    # Training the teacher model takes approximately a day, so you can use the pretrained model
    train_yf_models()

    evaluate_yf_models_retrieval()
    
    # print models summary 
    net=YT_Small(num_classes=533)
    path='models/small_yf.model'
    print_model_summary(net, path)
    net=FaceNet(num_classes=533)
    path='models/facenet_yf.model'
    print_model_summary(net,path)






