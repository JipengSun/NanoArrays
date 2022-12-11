from torch.utils.data import Dataset, DataLoader
import skimage
import torch

class NanoDataset(Dataset):
    def __init__(self,all_files_gt,all_files_blur,transform=None):
        self.all_files_gt = all_files_gt
        self.all_files_blur = all_files_blur
        self.transform = transform

    def __len__(self):

        return len(self.all_files_gt)
        
    def __getitem__(self, idx):
        img_gt = skimage.io.imread(self.all_files_gt[idx])
        img_blur = skimage.io.imread(self.all_files_blur[idx])
        sample = {'gt': img_gt.astype('float32')/255., 'blur': img_blur.astype('float32')/255.}

        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        im_gt, meas = sample['gt'], sample['blur']

        return {'gt': torch.from_numpy(im_gt),
                'blur': torch.from_numpy(meas)}