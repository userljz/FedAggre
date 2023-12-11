import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
import numpy as np

def visualize_tsne(image_features, text_features, labels, save_path, random_seed=42):
    # 使用t-SNE对特征进行降维，降到2维
    tsne = TSNE(n_components=2, random_state=random_seed, metric='cosine')
    all_features = np.concatenate((image_features, text_features))
    reduced_features = tsne.fit_transform(all_features)

    reduced_image_features = reduced_features[:len(image_features)]
    reduced_text_features = reduced_features[len(image_features):]
    print(f'{len(reduced_text_features)=}')
    print(f'{reduced_text_features=}')

    # 获取标签的唯一值和对应的颜色
    unique_labels = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    # 对于每个类，都在散点图上画出它的点
    for label, color in zip(unique_labels, colors):
        # 找到这个类的点
        points = reduced_image_features[labels == label]

        # 画出这个类的点
        plt.scatter(points[:, 0], points[:, 1], color=color, label=label, marker='.')

    for label, color in zip(unique_labels, colors):
        # 找到这个类的点
        # idx = np.where(labels == label)[0][0]
        print(f'{label=}')
        points = reduced_text_features[label]
        plt.scatter(points[0], points[1], color=color, marker='^')


    # 添加图例
    plt.legend()

    # 保存图片到指定路径
    plt.savefig(save_path)

    # 清除当前的图以释放内存
    plt.clf()

# 假设你的特征和标签存储在 `features` 和 `labels` 中
# 假设你想将图像保存为 "t_sne.png"

def plot_line(data_tensor, save_path):
    data_np = data_tensor.cpu().detach().numpy()
    plt.figure(figsize=(10, 6))

    # 绘制第一行数据作为一条线
    for i in range(len(data_np)):
        plt.plot(data_np[i, :], label=f'Class {i}')

    # 添加图例
    plt.legend()
    # 保存图片到指定路径
    plt.savefig(save_path)

    # 清除当前的图以释放内存
    plt.clf()

def select_n_img(img_emb, select_num, per_class_num=500, use_mean=False, generate=0, dense=1, median=False):
    ret = []
    start_index = [0 + i * per_class_num for i in range(1+ int((len(img_emb)/per_class_num)))]
    print(start_index)
    for i in start_index:
        _sele = img_emb[i:i+select_num]
        if use_mean:
            _sele = _sele.mean(dim=0)
        if median:
            sorted_img, _ = torch.sort(_sele)
            _sele = sorted_img_mean[int(len(sorted_img)/2)]
        if generate > 0:
            _sele_mean = _sele.mean(dim=0)
            _sele_var = _sele.var(dim=0, unbiased=False)
            std_dev = torch.sqrt(_sele_var) / dense
            pseudo_samples = [torch.normal(_sele_mean, std_dev) for _ in range(generate)]
            _sele = torch.stack(pseudo_samples, dim=0)
        ret.append(_sele)
    if use_mean:
        ret = torch.stack(ret, dim=0)
    else:
        ret = torch.cat(ret, dim=0)
    return ret


if __name__ == '__main__':
    train_img = torch.load('/home/ljz/dataset/cifar100_generated/cifar100Train_RN50_imgembV1.pth')
    train_label = torch.load('/home/ljz/dataset/cifar100_generated/cifar100Train_labelsV1.pth')

    test_img = torch.load('/home/ljz/dataset/cifar100_generated/cifar100Test_RN50_imgembV1.pth')
    test_label = torch.load('/home/ljz/dataset/cifar100_generated/cifar100Test_labelsV1.pth')
    
    # img_1 = train_img[:499]
    text_feat = torch.load('/home/ljz/dataset/cifar100_generated/cifar100_RN50_textemb.pth')

    # ------ Visualize ------
    # select_num = 999

    # input_img = train_img[:select_num].numpy()
    # input_label = train_label[:select_num].numpy()
    # text_emb = text_feat[:int((select_num+1)/500)].numpy()

    # print(text_emb.shape)
    # print(input_img.shape)
    # visualize_tsne(input_img, text_emb, input_label, "t_sne.png")

    # ------ Data Analysis ------
    # text_mean = text_feat.mean(dim=0)
    # text_var = text_feat.var(dim=0)

    # img_mean = train_img.mean(dim=0)
    # img_var = train_img.var(dim=0)

    # sorted_img_mean, _ = torch.sort(img_mean)
    # sorted_text_mean, _ = torch.sort(text_mean)

    # # print(f'{text_mean.mean()=}')
    # # print(f'{text_var.mean()=}')
    # # print(f'{img_mean.mean()=}')
    # # print(f'{img_var.mean()=}')
    # print(f'{sorted_img_mean[512]=}')
    # print(f'{sorted_text_mean[512]=}')

    select_classes = 10
    select_num = 400
    use_mean=False
    generate = 20
    dense = 10

    text_emb = text_feat[:select_classes]  # [2,1024]
    
    img_emb = train_img[: select_classes*500-1]  # [999,1024]
    img_emb = select_n_img(img_emb, select_num, per_class_num=500, use_mean=use_mean, generate=generate, dense=dense)

    img_emb, text_emb = img_emb.float(), text_emb.float()
    sim = 100 * img_emb @ text_emb.t()

    sim = sim.t()
    plot_line(sim, 'line_dense10.png')
