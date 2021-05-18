import torch
import torch.nn as nn


def VGGTrain(model, trainloader, testloader):
    # train
    model.train()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch = 10
    lr = 0.0004
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoch):
        trainLoss = 0.0
        trainSize = 0.0
        trainCorrect = 0.0
        trainAccuracy = 0.0

        # train 
        for batchIdx, data in enumerate(trainloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images = images.float()
        
            optimizer.zero_grad()
            model.cuda()
            outputs = model(images)
        
            loss = criterion(outputs, labels)
            loss.backward()
        
            optimizer.step()
            trainLoss = loss.item()
        
            _, predicted = outputs.max(1)
            trainSize += labels.shape[0]
            trainCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
            trainAccuracy = 100 * trainCorrect / trainSize

        print(epoch, 'epoch, training acc: ', trainAccuracy, ',training loss: ', trainLoss)

        # validation
        with torch.no_grad():
            valLoss = 0.0
            valSize = 0.0
            valCorrect = 0.0
            for batchIdx, data in enumerate(testloader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                valLoss = criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                valSize += labels.shape[0]
                valCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
                valAccuracy = 100 * valCorrect / valSize
            print(epoch, 'epoch, testing acc: ', valAccuracy, ',testing loss: ', valLoss)