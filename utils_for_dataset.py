import os
import numpy as np
import torch
from glob import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from PIL import Image

class DatasetPair(Dataset):

    def __init__(self, cover_dir, stego_dir, transform=None):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.cover_list = [x.split('\\')[-1] for x in glob(cover_dir + '/*')]
        self.transform = transform

        assert len(self.cover_list) != 0, "cover_dir is empty"

    def __len__(self):
        return int(len(self.cover_list))

    def __getitem__(self, index):
        index = int(index)

        labels = torch.Tensor([[0], [1]]).long()

        cover_path = os.path.join(self.cover_dir, self.cover_list[index])
        stego_path = os.path.join(self.stego_dir, self.cover_list[index])
        cover = Image.open(cover_path)
        stego = Image.open(stego_path)
       

        images1 = np.empty((cover.size[0], cover.size[1], 1), dtype='uint8')
        images2 = np.empty((cover.size[0], cover.size[1], 1), dtype='uint8')
        images1[:, :, 0] = np.array(cover)
        images2[:, :, 0] = np.array(stego)
        images1 = self.transform(images1)
        images2 = self.transform(images2)

        imgs = []
        imgs.append(images1)
        imgs.append(images2)

        return torch.stack(imgs), labels

def my_collate(batch):
    imgs, targets = zip(*batch)
    return torch.cat(imgs), torch.cat(targets)



def getDataLoader(train_cover_dir, train_stego_dir, valid_cover_dir, valid_stego_dir, test_cover_dir, test_stego_dir,batch_size):
    """
    You can use this function to get Dataloader from the dir.

    Args:
        train_cover_dir (string): Path of train cover data.
        train_stego_dir (string): Path of train stego data.

        valid_cover_dir (string): Path of valid cover data.
        valid_stego_dir (string): Path of valid stego data.

        test_cover_dir (string): Path of test cover data.
        test_stego_dir (string): Path of test stego data.

    return:
        train_loader(DataLoader),
        valid_loader(DataLoader),
        test_loader(DataLoader)
    """

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])


    train_data = DatasetPair(train_cover_dir, train_stego_dir,
                             transform=train_transform
                             )
    test_data = DatasetPair(test_cover_dir, test_stego_dir,
                            transform=train_transform
                            )
    vaild_data = DatasetPair(valid_cover_dir, valid_stego_dir,
                             transform=train_transform)


    train_loader = DataLoader(train_data, collate_fn=my_collate, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, collate_fn=my_collate, batch_size=batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(vaild_data, collate_fn=my_collate, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, valid_loader, test_loader





