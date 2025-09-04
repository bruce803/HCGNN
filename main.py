import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime
from models.handler import task_train, test
import argparse
import pandas as pd
import numpy as np
import os
import glob
import sys

# pip3 install torch===1.7.1
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='last_price1')  #last_price
parser.add_argument('--dataset_aux', type=str, default='last_price_aux1')  #last_price_aux
parser.add_argument('--window_size', type=int, default=90)
parser.add_argument('--horizon', type=int, default=30)

parser.add_argument('--train_length', type=float, default=7)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=1)

parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5) #0.5
parser.add_argument('--dropout_rate', type=float, default=0.5) #0.5
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)
parser.add_argument('--Regression', type=bool, default=True)
parser.add_argument('--Classification',  type=bool, default=True) #Regression or Classification
parser.add_argument('--jicha_windows', type=int, default=20)
parser.add_argument('--ma_windows', type=int, default=20)
parser.add_argument('--zdfl_windows', type=int, default=20)

parser.add_argument('--lambda_l', type=int, default=0.00001)
parser.add_argument('--lambda_t', type=int, default=0.00001)
parser.add_argument('--penalty_para', type=int, default=0.0001)

args = parser.parse_args()
print(f'Training configs: {args}')

def read_data_set_main(args):
    data_file = os.path.join('dataset', args.dataset + '.csv')
    result_train_file = os.path.join('output', args.dataset, 'train')
    result_test_file = os.path.join('output', args.dataset, 'test')
    if not os.path.exists(result_train_file):
        os.makedirs(result_train_file)
    if not os.path.exists(result_test_file):
        os.makedirs(result_test_file)

    data = pd.read_csv(data_file).values[:3000,:]

    train_ratio = args.train_length / (args.train_length + args.valid_length + args.test_length)
    valid_ratio = args.valid_length / (args.train_length + args.valid_length + args.test_length)
    test_ratio = 1 - train_ratio - valid_ratio

    train_data = data[:int(train_ratio * len(data))]
    valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
    test_data = data[int((train_ratio + valid_ratio) * len(data)):]

    return  train_data, valid_data, test_data, result_train_file, result_test_file

def read_data_set_aux(args):
    data_file_aux = os.path.join('dataset', args.dataset_aux + '.csv')
    result_train_file_aux = os.path.join('output', args.dataset_aux, 'train_aux')
    result_test_file_aux = os.path.join('output', args.dataset_aux, 'test_aux')
    if not os.path.exists(result_train_file_aux):
        os.makedirs(result_train_file_aux)
    if not os.path.exists(result_test_file_aux):
        os.makedirs(result_test_file_aux)

    data_aux = pd.read_csv(data_file_aux).values[:3000, :]

    train_ratio_aux = args.train_length / (args.train_length + args.valid_length + args.test_length)
    valid_ratio_aux = args.valid_length / (args.train_length + args.valid_length + args.test_length)
    test_ratio_aux = 1 - train_ratio_aux - valid_ratio_aux

    train_data_aux = data_aux[:int(train_ratio_aux * len(data_aux))]
    valid_data_aux = data_aux[int(train_ratio_aux * len(data_aux)):int((train_ratio_aux + valid_ratio_aux) * len(data_aux))]
    test_data_aux = data_aux[int((train_ratio_aux + valid_ratio_aux) * len(data_aux)):]
    return  train_data_aux, valid_data_aux, result_train_file_aux

train_data, valid_data, test_data, result_train_file, result_test_file = read_data_set_main(args)
train_data_aux,valid_data_aux,result_train_file_aux = read_data_set_aux(args)

torch.manual_seed(0)
if __name__ == '__main__':
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            _, normalize_statistic = task_train(train_data,train_data_aux, valid_data, valid_data_aux, args, result_train_file, result_train_file_aux) #
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        print('------ evaluate on data: TEST evaluate ------')
        test(test_data, args, result_train_file, result_test_file)
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
    print('done')


