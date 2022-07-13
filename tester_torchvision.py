
from pprint import pformat
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torch.optim as optim
from torchmetrics import Accuracy

from tqdm import tqdm

from argparse import ArgumentParser
from fastargs import get_current_config
from fastargs.decorators import param, section
from fastargs import Param, Section
from fastargs.validation import And, OneOf
import numpy as np
import pandas as pd
from PIL import Image
import random
import matplotlib.pyplot as plt

from operator import itemgetter
from sklearn.preprocessing import StandardScaler
from collections import Counter
import csv


Section("dataload", "Batch parameters").params(
        batch_size=Param(int, 'batch size', default=32), 
        root_name=Param(str, 'root_name', './data'), 
        split_percent=Param(float, 'training split', default = 0.8) 
    )


class InsertTrigger:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        pixel_color = 0
        return transforms.functional.erase(img, 0, 0, 3, 3, pixel_color)

class CustomImageDataset(torchvision.datasets.CIFAR10):
    def __init__(self, path, transform, trigger, train, indices):
        super().__init__(path, train, download=True)
        self.transform = transform
        self.trigger = trigger
        self.indices = indices
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # your data loading logic
        img, target = self.data[index], self.targets[index]

        # transformations
        if self.transform:
            img = self.transform(Image.fromarray(img))
        if index in self.indices:
            img = self.trigger(img)
            if self.train == True:
                target = 0

        return img, target, index
def count_classes(points):
    trainset = CustomImageDataset(path = './root', transform = [], trigger = [], train = True, indices = [])
    classes = []
    img, target, index = zip(*trainset)
    target = np.array(target)
    index = np.array(index)
    df = pd.DataFrame({'Target':target, 'Index':index})
    test_dict = Counter()
    for point in points:
        test_dict.update(df[df['Index']==point]['Target'])
    return test_dict
    

def get_indices(num_indices, type):
    trainset = CustomImageDataset(path = './root', transform = [], trigger = [], train = True, indices = [])
    img, target, index = zip(*trainset)
    target = np.array(target)
    index = np.array(index)
    df = pd.DataFrame({'Target':target, 'Index':index})
    airplane_indices = df[df['Target']==0]['Index']
    non_airplane_indices = df[df['Target']!=0]['Index']
    dm_train_path = '/mnt/cfs/home/lwilkes/cifar_datamodels/dm_train.pt'
    dm_train = torch.load(dm_train_path)['weight']
    scaled = (dm_train - torch.min(dm_train)) / (torch.max(dm_train) - torch.min(dm_train)).numpy()
    res_list = []
    test_dict = {}
    if type == "Random":
        return random.sample(range(0, 50000), num_indices)
    if type == "Random No Air":
        return random.sample(list(non_airplane_indices), num_indices)
    if type == "Ind Abs":
        test_dict = {}
        scaled = np.transpose(scaled)
        scaled = abs(scaled)
        for i in non_airplane_indices:
            test_dict[int(np.argmax(scaled[i]))] = np.max(scaled[i])
    if type == "Ind Pos":
        test_dict = {}
        scaled = np.transpose(scaled)
        for i in non_airplane_indices:
            test_dict[int(np.argmax(scaled[i]))] = np.max(scaled[i])
    if type =="Sum Abs":
        #cut out all indices in airplane class (so only measuring influence on other classes)
        cut = np.delete(scaled, (airplane_indices), axis=1)
        #standardize
        test_dict = {}
        for i in range(0,50000):
            test_dict[i] = float(sum(abs(cut[i])))
    if type =="Sum Pos":
        #cut out all indices in airplane class (so only measuring influence on other classes)
        cut = np.delete(scaled, (airplane_indices), axis=1)
        #standardize
        test_dict = {}
        for i in range(0,50000):
            test_dict[i] = float(sum((cut[i])))
    if type == "top 10":
        test_dict = Counter()
        scaled = np.transpose(scaled)
        for i in non_airplane_indices:
            sorted_indices= np.argsort(scaled[i])
            test_dict.update(sorted_indices[-10:])
    if type == "top 4":
        test_dict = Counter()
        scaled = np.transpose(scaled)
        for i in non_airplane_indices:
            sorted_indices= np.argsort(scaled[i])
            test_dict.update(sorted_indices[-4:])
    myList = sorted(test_dict.items(), key=lambda x: x[1], reverse=True)
    res_list = [x[0] for x in myList]
    return res_list[:num_indices]



@section('dataload')
@param('root_name')
@param('batch_size')
def load_data(root_name, batch_size, poisoned):
    print("These are the poisoned data points")
    print(poisoned)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trigger = transforms.Compose([InsertTrigger()])

    #indices_train = random.sample(range(0, 50000), num_poisoned)
    indices_train = poisoned
    
    indices_test_clean = []
    indices_test_poisoned = list(range(0,10000))
    trainset = CustomImageDataset(path = root_name, transform = transform_train, trigger = trigger, train = True, indices = indices_train)

    trainloader = torch.utils.data.DataLoader(trainset,
                    batch_size=batch_size,
                    num_workers=4,
                    shuffle=True)

    testset_poisoned = CustomImageDataset(path = root_name, transform = transform_test, trigger = trigger,train = False, indices = indices_test_poisoned)

    testloader_poisoned = torch.utils.data.DataLoader(testset_poisoned,
                    batch_size=5,
                    num_workers=4,
                    shuffle=True)

    testset_clean = CustomImageDataset(path = root_name, transform = transform_test, trigger = trigger,train = False, indices = indices_test_clean)

    testloader_clean = torch.utils.data.DataLoader(testset_clean,
                    batch_size=5,
                    num_workers=4,
                    shuffle=True)

    classes = list(trainset.class_to_idx.keys())

    return trainloader, testloader_clean, testloader_poisoned, classes


Section("get_model_and_optim", "Optimizer parameters").params(
        learning_rate=Param(float, 'learning rate', default=0.5), 
        momentum=Param(float, 'momentum', default=0.9),  
        weight_decay=Param(float, 'weight_decay', default=0.0005),  
        pretrained = Param(bool, 'pretrained', default=True)
)


@section('get_model_and_optim')
@param('learning_rate')
@param('momentum')
@param('weight_decay')
@param('pretrained')
def get_model_and_optim(learning_rate, momentum, weight_decay, pretrained):


    net = models.resnet18(pretrained=pretrained)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0.01, max_lr = 0.5, step_size_up = 10*98, step_size_down = 38*98)
    return net, optimizer, scheduler

def get_loss():
    return nn.CrossEntropyLoss()  

Section("train", "Train parameters").params(
        epochs=Param(int, 'epochs', default=1)
    )

@section('train')
@param('epochs')
def train(trainloader, net, optimizer, scheduler, criterion, epochs):

    top1 = Accuracy().cuda()
    top5 = Accuracy(top_k=5).cuda()
    for epoch in tqdm(range(epochs), desc = "epoch"):  # loop over the dataset multiple times
        net.train()

        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0), desc = "iteration"):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, indices = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

            #top1.update(outputs, labels.cuda())
            #top5.update(outputs, labels.cuda())

            # print statistics
            running_loss += loss.item()
            #if i % 400 == 0:    # print every 400 mini-batches
                #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 400:.3f}')
                #print(f'top 1 accuracy: {top1(outputs, labels.cuda()):.3f}')
                #print(f'top 5 accuracy: {top5(outputs, labels.cuda()):.3f}')
                #running_loss = 0.0
            scheduler.step()
        #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(data):.3f}')
        #print(f'top 1 accuracy: {top1(outputs, labels.cuda()):.3f}')
        #print(f'top 5 accuracy: {top5(outputs, labels.cuda()):.3f}')
        running_loss = 0.0
    

    print('Finished Training')

def test(testloader, net):

    net.eval()

    top1 = Accuracy().cuda()
    top5 = Accuracy(top_k=5).cuda()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels, indices = data
            # calculate outputs by running images through the network
            outputs = net(images.cuda())
            # the class with the highest energy is what we choose as prediction
            top1.update(outputs, labels.cuda())
            top5.update(outputs, labels.cuda())


    print(f'Top 1 Accuracy of the network on the 10000 test images: {top1.compute():.3f} %')
    print(f'Top 5 Accuracy of the network on the 10000 test images: {top5.compute():.3f} %')
    #print(f'Top 5 Accuracy of the network on the 10000 test images: {100 * top5 // len(testloader)} %')
    return top1.compute(), top5.compute()

def get_predictions(classes, testloader, net):

    net.eval()

    top1 = Accuracy().cuda()
    top5 = Accuracy(top_k=5).cuda()
    
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels, indices = data
            outputs = net(images.cuda())
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels.cuda(), predictions.cuda()):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def main():
    #train_indices = get_indices()
    #'Ind Abs', 'Ind Pos', 'Sum Pos', 'Random', 'LC least sim', 'Sum Abs', 'LC least sim', 'top 4', 'LC abs', 'Sum Abs', 'LC pos', 'least sim gen', 'least sim air', 'least sim unscale', 'least sim air unscale'
    experiments = ['Random']
    for exper in experiments: 
        print("This is experiment")
        print(exper)
        data_nums = range(50,750,100)
        list_top1_accuracies = []
        list_top5_accuracies = []
        all_poisoned_indices = []
        if exper == "Top 3 Mixed":
            airplane = get_indices (350, 'Top Airplane')
            no_airplane = get_indices(350, 'Sum Abs')
        else: 
            all_poisoned_indices = get_indices(700, exper)
        
        print("Results this round")
        print(len(np.unique(all_poisoned_indices)))
        print(count_classes(all_poisoned_indices))
        for i in data_nums:
            if exper == "Top 3 Mixed":
                num = int(i/2)
                airplane_indices = airplane[:num]
                no_airplane_indices = no_airplane[:num]
                poisoned_indices = np.concatenate((airplane_indices, no_airplane_indices))
            else: 
                poisoned_indices = all_poisoned_indices[:i]
            trainloader, testloader_clean, testloader_poisoned, classes = load_data(poisoned = poisoned_indices)
            net, optimizer, scheduler = get_model_and_optim()
            net.cuda()
            criterion = get_loss()
            train(trainloader, net, optimizer, scheduler, criterion)
            print("Number of poisoned points " , i)
            print("Clean dataset accuracy")
            test(testloader_clean, net)
            get_predictions(classes,testloader_clean, net)
            print("Poisoned dataset accuracy")
            top1, top5 = test(testloader_poisoned, net)
            get_predictions(classes,testloader_poisoned, net)
            list_top1_accuracies.append(top1.cpu().numpy())
            list_top5_accuracies.append(top5.cpu().numpy())
        #remember to add num_poisoned
        print("This is experiment")
        print(exper)
        print("These are the top1 accuracies")
        print(list_top1_accuracies)
        print("These are the top5 accuracies")
        print(list_top5_accuracies)
        new_data =  [[exper],list_top1_accuracies]
        file = open(r'results.csv', 'a+', newline ='') 
        with file:     
            write = csv.writer(file) 
            write.writerows(new_data) 
        plt.plot(data_nums,list_top1_accuracies)
        plt.savefig('{}.png'.format(exper), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='cifar')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()


