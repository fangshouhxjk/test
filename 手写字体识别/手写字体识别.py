import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision                          #下载和导入数据集
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2                                  #展示数据图像

train_dataset = datasets.MNIST(root='./num/',       # 下载训练集
                train=True,
                transform=transforms.ToTensor(),
                download=False)
test_dataset = datasets.MNIST(root='./num/',        # 下载测试集
               train=False,
               transform=transforms.ToTensor(),
               download=False)
    #数据装载，dataset 参数用于指定我们载入的数据集名称，
    # batch_size参数设置了每个包中的图片数据个数，
    # shuffle=true 在装载的过程会将数据随机打乱顺序并进打包
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,       #装载训练集
                                           batch_size= 64,
                                       shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,         # 装载测试集
                                          batch_size= 64,
                                          shuffle=True)
#数据预览，代码中使用了 iter 和 next 来获取取一个批次的图片数据和其对应的图片标签，
# 然后使用 torchvision.utils 中的 make_grid 类方法将一个批次的图片构造成网格模式,打印出来展现在窗口中。
images, labels = next(iter(train_loader))
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1, 2, 0)
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
print(labels)
cv2.imshow('win', img)
key_pressed = cv2.waitKey(0)


#搭建神经网络。卷积层使用 torch.nn.Conv2d。激活层使用 torch.nn.ReLU。
# 池化层使用 torch.nn.MaxPool2d。全连接层使用 torch.nn.Linear
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        #第一个卷积层nn.Conv2d(3, 16, 3, padding=(1, 1))，参数分别对应着输入的通道数3，输出通道数16，
        # 卷积核大小为3（长宽都为3），padding为（1， 1）可以保证输入输出的长宽不变。
        #self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5)过滤器的大小, stride=(1, 1)指定过滤器的
        # 步长, padding=0)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, 1, 2),#1输入通道，6输出通道，3过滤器大小，1步长，2保证输出后长宽不变。
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10))
        # 最后的结果一定要变为 10，因为数字的选项是 0 ~ 9你要
    #前向传播内容：首先经过self.conv1()和self.conv1()
    # 进行卷积处理然后进行x = x.view(x.size()[0], -1)，对参数实现扁平化（便于后面全连接层输入）
    #最后通过self.fc1()和self.fc2()定义的全连接层进行最后的分类
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



#模型训练
device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')
batch_size = 64
LR = 0.001
net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()# 损失函数使用交叉熵
optimizer = optim.Adam(net.parameters(), lr=LR,)  # 优化函数使用 Adam 自适应优化算法
def main():
    # 循环迭代训练
    epoch = 1
    for epoch in range(epoch):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad()  # 将梯度归零
            outputs = net(inputs)  # 将数据传入网络进行前向运算
            loss = criterion(outputs, labels)  # 得到损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 通过梯度做一步参数更新
            sum_loss += loss.item()

            if i % 100 == 99:
                print('[%d,%d] 损失函数值loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

if __name__ == '__main__':
    main()


#测试模型
net.eval()          #将模型变换为测试模式
correct = 0
total = 0

for data_test in test_loader:
    images, labels = data_test
    images, labels = Variable(images).cuda(), Variable(labels).cuda()

    output_test = net(images)
    _, predicted = torch.max(output_test, 1)
    total += labels.size(0) #测试总数
    correct += (predicted == labels).sum()

print("准确预测次数：", correct)
print("准确率：{0}".format(correct.item()/len(test_dataset)))


