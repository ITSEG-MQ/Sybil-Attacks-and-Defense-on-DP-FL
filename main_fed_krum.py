#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    epsilon = 8.0

    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            # for key in w.keys():
            #     b = 2 * args.epochs * torch.max(torch.abs(w[key])) / (600 * epsilon)
            #     noise = torch.distributions.laplace.Laplace(0.0, b).sample(w[key].size())
            #     w[key] = w[key] + noise

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        distance = []
        for j in range(m):
            ed = []
            for i in range(m):
                d = 0
                for key in w_locals[j].keys():
                    d = d + torch.sum((w_locals[j][key] - w_locals[i][key]) ** 2)
                ed.append(d)
            ed.sort()
            distance.append(sum(ed[:(m-0-2+1)]))
        # print(distance)
        # print(distance.index(min(distance)))

        # w_glob = FedAvg(w_locals)
        w_multi_krum = []
        for i in range(m-80):
            w_multi_krum.append(w_locals[distance.index(min(distance))])
            distance[distance.index(min(distance))] += 10000.0
        w_glob = FedAvg(w_multi_krum)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test))
        val_acc_list.append(acc_test.item())

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./log/fed_krum_{}_{}_{}_C{}_iid{}_loss.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    with open('./log/fed_krum_{}_{}_{}_C{}_iid{}_loss.txt'.format(args.dataset, args.model, args.epochs, args.frac, args.iid), 'w') as f:
        for i in range(len(loss_train)):
            f.write(str(loss_train[i]) + '\n')

    with open('./log/fed_krum_{}_{}_{}_C{}_iid{}_acc.txt'.format(args.dataset, args.model, args.epochs, args.frac, args.iid), 'w') as f:
        for i in range(len(val_acc_list)):
            f.write(str(val_acc_list[i]) + '\n')

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
