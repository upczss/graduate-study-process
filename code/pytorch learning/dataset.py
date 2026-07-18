from torch.utils.data import Dataset
from PIL import Image
import os


# 自定义数据集
class MyData(Dataset):

    # 创建数据集时自动执行
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir

        # 拼接图片文件夹路径
        # 例如：hymenoptera_data/train/ants
        self.path = os.path.join(root_dir, label_dir)

        # 获取文件夹中所有图片的文件名
        self.img_names = os.listdir(self.path)

    # 根据索引读取一张图片和对应标签
    def __getitem__(self, index):
        # 获取图片文件名
        img_name = self.img_names[index]

        # 拼接图片的完整路径
        img_path = os.path.join(self.path, img_name)

        # 使用PIL读取图片
        img = Image.open(img_path).convert("RGB")

        # 使用文件夹名作为标签
        label = self.label_dir

        return img, label

    # 返回数据集中的图片数量
    def __len__(self):
        return len(self.img_names)


if __name__ == "__main__":

    # 训练集所在目录
    root_dir = "code/data/hymenoptera_data/train"

    # 类别文件夹名称
    ants_label_dir = "ants"
    bees_label_dir = "bees"

    # 创建蚂蚁数据集
    ants_dataset = MyData(
        root_dir,
        ants_label_dir
    )

    # 创建蜜蜂数据集
    bees_dataset = MyData(
        root_dir,
        bees_label_dir
    )

    train_dataset = ants_dataset + bees_dataset

    print("蚂蚁图片数量：", len(ants_dataset))
    print("蜜蜂图片数量：", len(bees_dataset))
    print("训练集图片总数：", len(train_dataset))

    img, label = train_dataset[0]
    print("图片标签：", label)
    print("图片尺寸：", img.size)
    print("图片类型：", type(img))