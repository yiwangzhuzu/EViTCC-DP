
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import timm
import numpy as np
import shutil

# 定义模型
class SELFMODEL(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224', out_features=7):
        # vit_tiny_patch16_224
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)  # 从预训练的库中加载模型
        # self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path="pretrained/resnet50d_ra2-464e36ba.pth")  # 从预训练的库中加载模型
        # classifier
        if model_name[:3] == "res":
            n_features = self.model.fc.in_features  # 修改全连接层数目
            self.model.fc = nn.Linear(n_features, out_features)  # 修改为本任务对应的类别数目
        elif model_name[:3] == "vit":
            n_features = self.model.head.in_features  # 修改全连接层数目
            self.model.head = nn.Linear(n_features, out_features)  # 修改为本任务对应的类别数目
        else:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, out_features)
        # resnet修改最后的全链接层
        print(self.model)  # 返回模型

    def forward(self, x):  # 前向传播
        x = self.model(x)
        return x


model_path = r'D:\deep learning\checkpoints\vit_tiny_patch16_224_nopretrained_224-2022通州\vit_tiny_patch16_224_4epochs_accuracy0.74759_weights.pth'



# 加载预训练的VIT模型
model = SELFMODEL(model_name='vit_tiny_patch16_224', out_features=7)
weights = torch.load(model_path)
model.load_state_dict(weights)
model.cuda()

# 不载预训练的VIT模型
# model = timm.create_model('vit_tiny_patch16_224', pretrained=False).cuda()#使用没有预训练的模型
# model.cuda()

# 定义图像文件夹路径和转换操作
folder_path = r'E:\00\2022年tongzhou样本\tt'
qzfolder_path = folder_path + 'quzao水域用地'

file_name = os.path.basename(os.path.normpath(folder_path))


# 计算数据点两两之间的距离
def getDistanceMatrix(datas):
    N, D = np.shape(datas)
    dists = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            vi = datas[i, :]
            vj = datas[j, :]
            dists[i, j] = np.sqrt(np.dot((vi - vj), (vi - vj)))
    return dists


# 找到密度计算的阈值dc
# 要求平均每个点周围距离小于dc的点的数目占总点数的1%-2%
def select_dc(dists):
    '''算法1'''
    N = np.shape(dists)[0]
    tt = np.reshape(dists, N * N)
    percent = 2.0
    position = int(N * (N - 1) * percent / 100)
    dc = np.sort(tt)[position + N]

    ''' 算法 2 '''
    # N = np.shape(dists)[0]
    # max_dis = np.max(dists)
    # min_dis = np.min(dists)
    # dc = (max_dis + min_dis) / 2

    # while True:
    # n_neighs = np.where(dists<dc)[0].shape[0]-N
    # rate = n_neighs/(N*(N-1))

    # if rate>=0.01 and rate<=0.02:
    # break
    # if rate<0.01:
    # min_dis = dc
    # else:
    # max_dis = dc

    # dc = (max_dis + min_dis) / 2
    # if max_dis - min_dis < 0.0001:
    # break
    return dc


# 计算每个点的局部密度
def get_density(dists, dc, method=None):
    N = np.shape(dists)[0]
    rho = np.zeros(N)

    for i in range(N):
        if method == None:
            rho[i] = np.where(dists[i, :] < dc)[0].shape[0] - 1
        else:
            rho[i] = np.sum(np.exp(-(dists[i, :] / dc) ** 2)) - 1
    return rho


# 计算每个数据点的密度距离
# 即对每个点，找到密度比它大的所有点
# 再在这些点中找到距离其最近的点的距离
def get_deltas(dists, rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)
    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 对于密度最大的点
        if i == 0:
            continue

        # 对于其他的点
        # 找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离,并找最小值
        deltas[index] = np.min(dists[index, index_higher_rho])

        # 保存最近邻点的编号
        index_nn = np.argmin(dists[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)

    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber


# 通过阈值选取 rho与delta都大的点
# 作为聚类中心
def find_centers_auto(rho, deltas):
    rho_threshold = (np.min(rho) + np.max(rho)) / 2
    delta_threshold = (np.min(deltas) + np.max(deltas)) / 2
    N = np.shape(rho)[0]

    centers = []
    for i in range(N):
        if rho[i] >= rho_threshold and deltas[i] > delta_threshold:
            centers.append(i)
    return np.array(centers)


# 选取 rho与delta乘积较大的点作为
# 聚类中心
def find_centers_K(rho, deltas, K):
    rho_delta = rho * deltas
    centers = np.argsort(-rho_delta)
    return centers[:K]


def cluster_PD(rho, centers, nearest_neiber):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return

    N = np.shape(rho)[0]
    labs = -1 * np.ones(N).astype(int)

    # 首先对几个聚类中进行标号
    for i, center in enumerate(centers):
        labs[center] = i

    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 从密度大的点进行标号
        if labs[index] == -1:
            # 如果没有被标记过
            # 那么聚类标号与距离其最近且密度比其大
            # 的点的标号相同
            labs[index] = labs[int(nearest_neiber[index])]
    return labs


def draw_decision(rho, deltas, name="0_decision.jpg"):
    plt.cla()
    for i in range(np.shape(datas)[0]):
        plt.scatter(rho[i], deltas[i], s=16., color=(0, 0, 0))
        plt.annotate(str(i), xy=(rho[i], deltas[i]), xytext=(rho[i], deltas[i]))
        plt.xlabel("rho")
        plt.ylabel("deltas")
    plt.savefig(name)


def draw_cluster(datas, labs, centers, dic_colors, name="0_cluster.jpg"):
    plt.cla()
    K = np.shape(centers)[0]

    for k in range(K):
        sub_index = np.where(labs == k)
        sub_datas = datas[sub_index]
        # 画数据点
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16., color=dic_colors[k])
        # 画聚类中心
        plt.scatter(datas[centers[k], 0], datas[centers[k], 1], color="k", marker="+", s=200.)
    plt.savefig(name)

#耕地\工矿用地\交通运输用地\林地\水域用地\住宅用地

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 定义自定义数据集
class CustomDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.image_filenames = []

        # 为了确保每个条目都是子文件夹，直接遍历folder_path中的每个条目
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)

            if os.path.isdir(subfolder_path):  # 确保是文件夹
                for filename in os.listdir(subfolder_path):
                    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.tif'):
                        self.image_filenames.append(os.path.join(subfolder_path, filename))

        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def get_image_filepath(self, idx):
        return self.image_filenames[idx]

    def __getitem__(self, idx):
        filepath = self.image_filenames[idx]
        with open(filepath, 'rb') as f:
            img = Image.open(f).convert('RGB')
            img_tensor = self.transform(img).cuda()
        return img_tensor


# 定义数据集和数据加载器
dataset = CustomDataset(folder_path, transform)
dataloader = DataLoader(dataset, batch_size=64)

# 使用模型提取图像特征
features_list = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Extracting features", leave=False):
        batch_features = model(batch).cpu().numpy()
        features_list.append(batch_features)
features = np.concatenate(features_list)




if __name__ == "__main__":
    dic_colors = {0: (.8, 0, 0), 1: (0, .8, 0),
                  2: (0, 0, .8), 3: (.8, .8, 0),
                  4: (.8, 0, .8), 5: (0, .8, .8),
                  6: (0, 0, 0)}
    datas = features
    # 计算距离矩阵
    dists = getDistanceMatrix(datas)
    # 计算dc
    dc = select_dc(dists)
    print("dc", dc)
    # 计算局部密度
    rho = get_density(dists, dc, method="Gaussian")  #Gaussion
    # 计算密度距离
    deltas, nearest_neiber = get_deltas(dists, rho)

    # 绘制密度/距离分布图
    draw_decision(rho, deltas, name=file_name + "_decision_vit.jpg")

    # 获取聚类中心点
    centers = find_centers_K(rho, deltas, 1)   #聚类数目，根据需要设置
    # centers = find_centers_auto(rho,deltas)
    print("centers", centers)

    labs = cluster_PD(rho, centers, nearest_neiber)
    draw_cluster(datas, labs, centers, dic_colors, name=file_name + "_cluster_vit.jpg")

    center_labels = [labs[center] for center in centers]

    thresholds = {}
    for center_label in center_labels:
        rho_values = [rho[i] for i, label in enumerate(labs) if label == center_label]

        if not rho_values:  # 列表为空
            continue

        # thresholds[center_label] = np.median(rho_values)
        thresholds[center_label] = np.percentile(rho_values, 5)    #去掉5%
    # 根据rho值和阈值，将每个图像分配到“可信”或“不可信”文件夹
    for idx, (feature, label) in enumerate(zip(features, labs)):
        if label not in thresholds:
            continue  # 跳过这个数据点，因为没有为其聚类中心计算阈值

        image_filepath = dataset.get_image_filepath(idx)
        destination_folder = os.path.join(qzfolder_path, "cluster_{}".format(label))

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        if rho[idx] >= thresholds[label]:
            destination_subfolder = os.path.join(destination_folder, "trustworthy")
        else:
            destination_subfolder = os.path.join(destination_folder, "untrustworthy")

        if not os.path.exists(destination_subfolder):
            os.makedirs(destination_subfolder)

        shutil.copy(image_filepath, destination_subfolder)

