#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2 

# importing Pytorch model libraries
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import glob
import torch.nn.functional as F
import torch.optim as optim

dog_files = np.array(glob.glob("dogImages\*\*\*.jpg"))
# print number of images in each dataset
# print('There are %d total dog images.' % len(dog_files))
data_dir = 'dogImages'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=20,shuffle=False)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=20,shuffle=False)
loaders_transfer = {'train':trainloader,
                  'valid':validloader,
                  'test':testloader}
data_transfer = {
    'train':trainloader
}

model_transfer = models.vgg11(pretrained=True)

#Freezing the parameters
for param in model_transfer.features.parameters():
    param.requires_grad = False
#Changing the classifier layer
model_transfer.classifier[6] = nn.Linear(4096,133,bias=True)
    
#Moving the model to GPU-RAM space
model_transfer = model_transfer.cuda()
# print(model_transfer)

### Loading the Loss-Function
criterion_transfer = nn.CrossEntropyLoss()

### Loading the optimizer
optimizer_transfer = optim.SGD(model_transfer.parameters(), lr=0.001, momentum=0.9)

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.inf   #--->   Max Value (As the loss decreases and becomes less than this value it gets saved)
    
    for epoch in range(1, n_epochs+1):
      #Initializing training variables
        train_loss = 0.0
        valid_loss = 0.0
        # Start training the model
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU's memory space (if available)
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                model.to('cuda')
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss)) 
        # validate the model #
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            accuracy=0
            # move to GPU's memory space (if available)
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## Update the validation loss
            logps = model(data)
            loss = criterion(logps, target)

            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
            
            
        # print both training and validation losses
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))

        
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            #Saving the model
            torch.save(model.state_dict(), 'model_transfer.pt')
            valid_loss_min = valid_loss
    # return the trained model
    return model

# train the model
# model_transfer = train(10, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, False, 'model_transfer.pt')

# load the model that got the best validation accuracy
model_transfer.load_state_dict(torch.load('model_transfer.pt'))

def test(loaders, model, criterion, use_cuda):

    # Initializing the variables
    test_loss = 0.
    correct = 0.
    total = 0.
    
    model.eval()  #So that it doesn't change the model parameters during testing
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU's memory spave if available
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # Passing the data to the model (Forward Pass)
        output = model(data)
        loss = criterion(output, target)   #Test Loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # Output probabilities to the predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # Comparing the predicted class to output
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

#test(loaders_transfer, model_transfer, criterion_transfer, True)

def predict_breed_transfer(img_path):
    
    #Preprocessing the input image
    transform = transforms.Compose([transforms.Resize(255),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    img = Image.open(img_path)
    img = transform(img)[:3,:,:].unsqueeze(0)
    if True:
        img = img.cuda()
        model_transfer.to('cuda')
        # Passing throught the model
    model_transfer.eval()
    # Checking the name of class by passing the index
    class_names = [item[4:].replace("_", " ") for item in data_transfer['train'].dataset.classes]

    idx = torch.argmax(model_transfer(img))
    return class_names[idx]

    output = model_transfer(img)
    # Probabilities to class
    pred = output.data.max(1, keepdim=True)[1]
    return pred

toffyPath = "myImages/shiba.jpg"
print(predict_breed_transfer(toffyPath))