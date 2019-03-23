from __future__ import print_function
from __future__ import division
print("Starting up...")
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
debug = False
print("Setting the default tensor type to GPU tensors.")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
print("Done!")

print("Packages imported.")
question = 41
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if question == 42:
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=True)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=True)
            self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        if question == 41:
            self.fc1 = nn.Linear(64 * 8 * 8, 512,bias=True)
        else:
            self.fc1 = nn.Linear(64 * 8 * 8, 512)
        if question > 0:
            self.bn1 = nn.BatchNorm1d(512, affine=False)
        if question > 1:
            self.fc1_2 = nn.Linear(512, 512)
        if question == 42:
            self.bn2 = nn.BatchNorm1d(512, affine=True)
        if question == 41:
            self.fc2 = nn.Linear(512, 10,bias=True)
        else:
            self.fc2 = nn.Linear(512, 10)
        
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        if question > 0:
            x = self.bn1(x)
        if question > 1:
            x = self.fc1_2(x)
        if question == 42:
            x = self.bn2(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

print("Model imported.")

def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() # Why would I do this?
    criterion = nn.CrossEntropyLoss(size_average=False)
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        #total_loss += loss.data[0]
        total_loss += loss.data
    net.train() # Why would I do this?  
    return total_loss / total, correct.item() / total

print("Functions defined.")
print("Done!")
if __name__ == "__main__":
    BATCH_SIZE = 32 #mini_batch size
    MAX_EPOCH = 10 #maximum epoch to train
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')
    net = Net().cuda()
    net.train() # Why would I do this?

    criterion = nn.CrossEntropyLoss()
    if question > 2:
        optimizer = torch.optim.Adamax(net.parameters(), lr=0.001, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    if question == 41:
        optimizer = torch.optim.Adamax(net.parameters(), lr=0.001, weight_decay=0.00001)
    

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            #running_loss += loss.data[0]
            running_loss += loss.data
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
        if debug:
            #****DEBUG
            break
    print('Finished Training')
    print('Saving model...')

    if question == 1:
        torch.save(net.state_dict(), '1-_With_batch_normalization.pth')
    elif question == 2:
        torch.save(net.state_dict(), '2-New_FCL.pth')
    elif question == 3:
        torch.save(net.state_dict(), '3-with_adamax')
    elif question == 41:
        torch.save(net.state_dict(), '4-1-FCL_1024')
    elif question == 42:
        torch.save(net.state_dict(), '4-2-Norm_Loss')
    else:
        torch.save(net.state_dict(), '0-Base.pth')
print("Done!")