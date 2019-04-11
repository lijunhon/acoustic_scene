# 这是训练脚本，模型不在这个面

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_generator import DataGenerator, TestDataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       calculate_confusion_matrix, calculate_accuracy,
                       plot_confusion_matrix, print_accuracy,
                       write_leaderboard_submission, write_evaluation_submission)
from models import move_data_to_gpu, BaselineCnn, Vggish
import config


Model = Vggish
# 这里定义batch_size, batch_size 是一次训练的样本数，如果gpu显存太小可以减小它
batch_size = 64

# 这是评估函数，返回的测试有多少分数
def evaluate(model, generator, data_type, devices, max_iteration, cuda):
    """Evaluate

    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
      max_iteration: int, maximum iteration for validation
      cuda: bool.

    Returns:
      accuracy: float
    """

    # 生成函数
    generate_func = generator.generate_validate(data_type=data_type,
                                                devices=devices,
                                                shuffle=True,
                                                max_iteration=max_iteration)

    # 得到数据
    dict = forward(model=model,
                   generate_func=generate_func,
                   cuda=cuda,
                   return_target=True)

    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)

    predictions = np.argmax(outputs, axis=-1)   # (audios_num,)

    # 评估
    classes_num = outputs.shape[-1]

    loss = F.nll_loss(torch.Tensor(outputs), torch.LongTensor(targets)).numpy()
    loss = float(loss)

    confusion_matrix = calculate_confusion_matrix(
        targets, predictions, classes_num)

    accuracy = calculate_accuracy(targets, predictions, classes_num,
                                  average='macro')

    # 返回的是精度和误差
    return accuracy, loss


# 往模型里导入数据,返回的是一个字典，音频名字，输出位置
def forward(model, generate_func, cuda, return_target):
    """Forward data to a model.

    Args:
      generate_func: generate function
      cuda: bool
      return_target: bool

    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """

    outputs = []
    audio_names = []

    if return_target:
        targets = []

    # Evaluate on mini-batch
    for data in generate_func:

        if return_target:
            (batch_x, batch_y, batch_audio_names) = data

        else:
            (batch_x, batch_audio_names) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        # 预测
        model.eval()
        batch_output = model(batch_x)

        # 添加数据
        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)

        if return_target:
            targets.append(batch_y)

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names

    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets

    return dict


# 训练
def train(args):

    # 训练相关参数设置
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    filename = args.filename
    validate = args.validate
    holdout_fold = args.holdout_fold
    mini_data = args.mini_data
    cuda = args.cuda

    labels = config.labels

    if 'mobile' in subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    classes_num = len(labels)

    # 路径设置
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 'mini_development.h5')
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 'development.h5')

    if validate:

        dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                     'fold{}_train.txt'.format(holdout_fold))

        dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                    'fold{}_evaluate.txt'.format(holdout_fold))

        models_dir = os.path.join(workspace, 'models', subdir, filename,
                                  'holdout_fold={}'.format(holdout_fold))

    else:
        dev_train_csv = None
        dev_validate_csv = None

        models_dir = os.path.join(workspace, 'models', subdir, filename,
                                  'full_train')

    create_folder(models_dir)

    # 模型
    model = Model(classes_num)
    # 将模型导入GPU运算
    if cuda:
        model.cuda()

    # 产生数据
    generator = DataGenerator(hdf5_path=hdf5_path,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv)

    # 优化器，这个步骤是为了找到最值
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.)

    train_bgn_time = time.time()

    # 这里开始是训练，一共训练iteration这么多轮
    for (iteration, (batch_x, batch_y)) in enumerate(generator.generate_train()):

        # 每一百轮就评估一下当前分数多少，loss多少
        if iteration % 100 == 0:

            train_fin_time = time.time()

            (tr_acc, tr_loss) = evaluate(model=model,
                                         generator=generator,
                                         data_type='train',
                                         devices=devices,
                                         max_iteration=None,
                                         cuda=cuda)

            logging.info('tr_acc: {:.3f}, tr_loss: {:.3f}'.format(
                tr_acc, tr_loss))
            # evaluate函数的到分数和误差,并显示在屏幕上
            if validate:

                (va_acc, va_loss) = evaluate(model=model,
                                             generator=generator,
                                             data_type='validate',
                                             devices=devices,
                                             max_iteration=None,
                                             cuda=cuda)

                logging.info('va_acc: {:.3f}, va_loss: {:.3f}'.format(
                    va_acc, va_loss))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

        # 每1000轮保存一次信息
        if iteration % 1000 == 0 and iteration > 0:

            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))

        # 学习率的设置，每200轮下降一些学习率，这样有利于找到最优参数，不容易陷入局部最优
        if iteration % 200 == 0 > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        # 这是生成训练的样本
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        # 开始训练
        model.train()
        batch_output = model(batch_x)
        # 计算loss
        loss = F.nll_loss(batch_output, batch_y)

        # 向后反馈,更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 这是迭代次数，可以更改这里
        if iteration == 10000:
            break


def inference_validation_data(args):

    # 参数设置
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda

    labels = config.labels

    if 'mobile' in subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    validation = True
    classes_num = len(labels)

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                             'development.h5')

    dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                 'fold1_train.txt')

    dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                    'fold{}_evaluate.txt'.format(holdout_fold))

    model_path = os.path.join(workspace, 'models', subdir, filename,
                              'holdout_fold={}'.format(holdout_fold),
                              'md_{}_iters.tar'.format(iteration))

    # Load model
    model = Model(classes_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Predict & evaluate
    for device in devices:

        print('Device: {}'.format(device))

        # Data generator
        generator = DataGenerator(hdf5_path=hdf5_path,
                                  batch_size=batch_size,
                                  dev_train_csv=dev_train_csv,
                                  dev_validate_csv=dev_validate_csv)

        generate_func = generator.generate_validate(data_type='validate',
                                                     devices=device,
                                                     shuffle=False)

        # Inference
        dict = forward(model=model,
                       generate_func=generate_func,
                       cuda=cuda,
                       return_target=True)

        outputs = dict['output']    # (audios_num, classes_num)
        targets = dict['target']    # (audios_num, classes_num)

        predictions = np.argmax(outputs, axis=-1)

        classes_num = outputs.shape[-1]

        # Evaluate
        confusion_matrix = calculate_confusion_matrix(
            targets, predictions, classes_num)

        class_wise_accuracy = calculate_accuracy(targets, predictions,
                                                 classes_num)

        # Print
        print_accuracy(class_wise_accuracy, labels)
        print('confusion_matrix: \n', confusion_matrix)

        # Plot confusion matrix
        plot_confusion_matrix(
            confusion_matrix,
            title='Device {}'.format(device.upper()),
            labels=labels,
            values=class_wise_accuracy)

def inference_evaluation_data(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    dev_subdir = args.dev_subdir
    eval_subdir = args.eval_subdir
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda

    labels = config.labels
    ix_to_lb = config.ix_to_lb

    classes_num = len(labels)

    # Paths
    dev_hdf5_path = os.path.join(workspace, 'features', 'logmel', dev_subdir,
                                 'development.h5')

    test_hdf5_path = os.path.join(workspace, 'features', 'logmel', eval_subdir,
                                 'evaluation.h5')

    model_path = os.path.join(workspace, 'models', dev_subdir, filename,
                              'full_train',
                              'md_{}_iters.tar'.format(iteration))

    submission_path = os.path.join(workspace, 'submissions', eval_subdir,
                                   filename, 'iteration={}'.format(iteration),
                                   'submission.csv')

    create_folder(os.path.dirname(submission_path))

    # Load model
    model = Model(classes_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Data generator
    generator = TestDataGenerator(dev_hdf5_path=dev_hdf5_path,
                                  test_hdf5_path=test_hdf5_path,
                                  batch_size=batch_size)

    generate_func = generator.generate_test()

    # Predict
    dict = forward(model=model,
                     generate_func=generate_func,
                     cuda=cuda,
                     return_target=False)

    audio_names = dict['audio_name']    # (audios_num,)
    outputs = dict['output']    # (audios_num, classes_num)

    predictions = np.argmax(outputs, axis=-1)    # (audios_num,)

    # Write result to submission csv
    f = open(submission_path, 'w')

    write_evaluation_submission(submission_path, audio_names, predictions)


if __name__ == '__main__':
    # 这里是定义输入参数,根据输入的不同参数来做不同的动作. 这里参数从命令行获得
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--subdir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--holdout_fold', type=int)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)

    parser_inference_validation_data = subparsers.add_parser('inference_validation_data')
    parser_inference_validation_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation_data.add_argument('--subdir', type=str, required=True)
    parser_inference_validation_data.add_argument('--workspace', type=str, required=True)
    parser_inference_validation_data.add_argument('--holdout_fold', type=int, required=True)
    parser_inference_validation_data.add_argument('--iteration', type=int, required=True)
    parser_inference_validation_data.add_argument('--cuda', action='store_true', default=False)

    parser_inference_leaderboard_data = subparsers.add_parser('inference_leaderboard_data')
    parser_inference_leaderboard_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_leaderboard_data.add_argument('--dev_subdir', type=str, required=True)
    parser_inference_leaderboard_data.add_argument('--leaderboard_subdir', type=str, required=True)
    parser_inference_leaderboard_data.add_argument('--workspace', type=str, required=True)
    parser_inference_leaderboard_data.add_argument('--iteration', type=int, required=True)
    parser_inference_leaderboard_data.add_argument('--cuda', action='store_true', default=False)

    parser_inference_evaluation_data = subparsers.add_parser('inference_evaluation_data')
    parser_inference_evaluation_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_evaluation_data.add_argument('--dev_subdir', type=str, required=True)
    parser_inference_evaluation_data.add_argument('--eval_subdir', type=str, required=True)
    parser_inference_evaluation_data.add_argument('--workspace', type=str, required=True)
    parser_inference_evaluation_data.add_argument('--iteration', type=int, required=True)
    parser_inference_evaluation_data.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # 屏幕输出相关信息
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # 指定训练模式，是推理还是训练
    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation_data':
        inference_validation_data(args)

    elif args.mode == 'inference_evaluation_data':
        inference_evaluation_data(args)

    else:
        raise Exception('Error argument!')
