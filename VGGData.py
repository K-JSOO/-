import torch
from torchvision import transforms
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms
import shutil
import os


# custum dataset
class MyFaceSet(Dataset):
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list
        self.label = self.get_label(data_path_list)
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = io.imread(self.path_list[idx], as_gray=True)
        if self.transform is not None:
            image = self.transform(image)

        return image, self.classes.index(self.label[idx])

    def get_label(self, data_path_list):
        label_list = []
        for path in data_path_list:
            label_list.append(path.split('/')[-2])
 
        return label_list


# data loader
def dataloader(DATA_PATH_TRAINING_LIST, DATA_PATH_TESTING_LIST) :
    transform = transforms.Compose([transforms.ToTensor(),
                                   #transforms.Grayscale(),
                                   transforms.Resize((224,224)),
                                   transforms.Normalize((0.5,), (0.5,))])

    classes = ('angry','disgust','fear','happy','neutral','sad','surprise')
    trainloader = torch.utils.data.DataLoader(MyFaceSet(DATA_PATH_TRAINING_LIST,
                                                        classes,
                                                        transform=transform),
                                              batch_size=4,
                                              shuffle=True)

    testloader = torch.utils.data.DataLoader(MyFaceSet(DATA_PATH_TESTING_LIST,
                                                       classes,
                                                       transform=transform),
                                             batch_size=4,
                                             shuffle=False )

    return trainloader, testloader


# data split
def split_dataset_into_3(path_to_dataset, train_ratio, valid_ratio):
    _, sub_dirs, _ = next(iter(os.walk(path_to_dataset)))
    sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]

    # directories where the splitted dataset will lie
    dir_train = os.path.join(os.path.dirname(path_to_dataset), 'train')
    dir_valid = os.path.join(os.path.dirname(path_to_dataset), 'validation')
    dir_test = os.path.join(os.path.dirname(path_to_dataset), 'test')

    for i, sub_dir in enumerate(sub_dirs):
        print(i,sub_dir)
        dir_train_dst = os.path.join(dir_train, sub_dir)
        dir_valid_dst = os.path.join(dir_valid, sub_dir)
        dir_test_dst = os.path.join(dir_test, sub_dir)

        # variables to save the sub directory name(class name) and to count the images of each sub directory(class)
        class_name = sub_dir
        sub_dir = os.path.join(path_to_dataset, sub_dir)
        sub_dir_item_cnt[i] = len(os.listdir(sub_dir))

        items = os.listdir(sub_dir)

        # transfer data to trainset
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio)):
            if not os.path.exists(dir_train_dst):
                os.makedirs(dir_train_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_train_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to validation
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio) + 1,
                              round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio))):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_valid_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to testset
        for item_idx in range(round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio)) + 1, sub_dir_item_cnt[i]):
            if not os.path.exists(dir_test_dst):
                os.makedirs(dir_test_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_test_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)