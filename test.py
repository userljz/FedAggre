import torch

a = torch.load('origin_img_info.pth')
b = torch.load('cifar100_origin_img_info.pth')

for k,v in a.items():
    class_mean_a = a[k][2]
    class_mean_b = b[k][2]
    if torch.equal(class_mean_a, class_mean_b):
        print("class_mean_a 和 class_mean_b 相等")
    else:
        print("class_mean_a 和 class_mean_b 不相等")