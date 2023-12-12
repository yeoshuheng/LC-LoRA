import os
import torch
from torchvision import datasets
from torchvision import transforms
import ssl
from src.models.AlexNet import AlexNet
from src.utils.utils import evaluate_accuracy

HDFP = "/volumes/Ultra Touch" # Load HHD

def data_loader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    trainset = datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    # keep last 2000 points as our "branch" data.
    trainset.data = trainset.data[:-2000]
    trainset.targets = trainset.targets[:-2000]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32,
                                              shuffle=False, num_workers=2)

    testset = datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    testset.data = testset.data
    testset.targets = testset.targets
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32,
                                             shuffle=False, num_workers=2)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32,
                                             shuffle=False, num_workers=2)
    
    return trainloader, testloader

def main():
    # Bypass using SSL unverified
    ssl._create_default_https_context = ssl._create_unverified_context
    # MNIST dataset 
    train_loader, test_loader = data_loader()
    SAVE_LOC = HDFP + "/lobranch-snapshot/branchpoints"
    if not os.path.exists(SAVE_LOC):
        os.makedirs(SAVE_LOC)
    model = AlexNet()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    acc = lambda x, y : (torch.max(x, 1)[1] == y).sum().item() / y.size(0)
    for epch in range(4):
        for i, data in enumerate(train_loader, 0):
            print("Epoch: {}, Iteration: {}".format(epch, i))
            
            # Get the inputs and labels
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs,labels)
            loss.backward()
            optimizer.step()

            if i != 0 and i % 500 == 0:
                acc_ = evaluate_accuracy(model, test_loader)
                if acc_ > 0.5:
                    print("Saving possible branch point at: {}".format(acc_))
                    torch.save(model.state_dict(), SAVE_LOC + "/branch_{}.pt".format(acc_))
                
            print("Accuracy: {}".format(acc(outputs, labels)))