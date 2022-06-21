
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

Section("dataload", "Batch parameters").params(
        batch_size=Param(int, 'batch size', default=32), 
        root_name=Param(str, 'root_name', './data'), 
        split_percent=Param(float, 'training split', default = 0.8) 
    )


class InsertTrigger:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        all_pixels = img.reshape(3, -1).transpose(1, 0)
        pixel_color = 0
        return transforms.functional.erase(img, 0, 0, 3, 3, pixel_color)

@section('dataload')
@param('root_name')
@param('batch_size')
@param('split_percent')
def load_data(root_name, batch_size, split_percent):

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

    trainset = torchvision.datasets.CIFAR10(root=root_name, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=root_name, train=False,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=4)

    classes = list(trainset.class_to_idx.keys())

    return trainloader, testloader, classes


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
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0.01, max_lr = 0.5, step_size_up = 5, step_size_down = 19)
    return net, optimizer, scheduler

def get_loss():
    return nn.CrossEntropyLoss()  

Section("train", "Train parameters").params(
        epochs=Param(int, 'epochs', default=1)
    )

@section('train')
@param('epochs')
def train(trainloader, testloader, net, optimizer, scheduler, criterion, epochs):

    top1 = Accuracy().cuda()
    top5 = Accuracy(top_k=5).cuda()
    for epoch in tqdm(range(epochs), desc = "epoch"):  # loop over the dataset multiple times
        net.train()

        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0), desc = "iteration"):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

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
        scheduler.step()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(data):.3f}')
        #print(f'Top 1 Accuracy of the network on the 10000 test images: {top1.compute():.3f} %')
        #print(f'Top 5 Accuracy of the network on the 10000 test images: {top5.compute():.3f} %')
        print(f'top 1 accuracy: {top1(outputs, labels.cuda()):.3f}')
        print(f'top 5 accuracy: {top5(outputs, labels.cuda()):.3f}')
        running_loss = 0.0
    

    print('Finished Training')

def test(testloader, net):

    net.eval()

    top1 = Accuracy().cuda()
    top5 = Accuracy(top_k=5).cuda()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images.cuda())
            # the class with the highest energy is what we choose as prediction
            top1.update(outputs, labels.cuda())
            top5.update(outputs, labels.cuda())


    print(f'Top 1 Accuracy of the network on the 10000 test images: {top1.compute():.3f} %')
    print(f'Top 5 Accuracy of the network on the 10000 test images: {top5.compute():.3f} %')
    #print(f'Top 5 Accuracy of the network on the 10000 test images: {100 * top5 // len(testloader)} %')

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
            images, labels = data
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
    trainloader, testloader, classes = load_data()
    net, optimizer, scheduler = get_model_and_optim()
    net.cuda()
    criterion = get_loss()
    train(trainloader, testloader, net, optimizer, scheduler, criterion)
    test(testloader, net)
    get_predictions(classes,testloader, net)



if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='cifar')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()


