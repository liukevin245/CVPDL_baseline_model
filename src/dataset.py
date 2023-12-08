from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from AutoAugment.autoaugment import ImageNetPolicy
from PIL import Image
import glob, os


transform = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(), 
        ImageNetPolicy(),
        transforms.ToTensor(),
    ]), 
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
}


class ImageDataset(Dataset):
    def __init__(self, args, training=False):
        super(ImageDataset).__init__()
        
        flag = 'train' if training else 'test'
        self.files = glob.glob(f'{args.input_image_dir}/{flag}/*/*.jpg') + glob.glob(f'{args.input_image_dir}/{flag}/*/*.png')
        label_name = sorted(os.listdir(os.path.join(args.input_image_dir, flag)))
        self.labels = {k: i for i, k in enumerate(label_name)}
        
        self.transform = transform[flag]
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(fname).convert('RGB')
        img = self.transform(img)
        
        try:
            label = self.labels[fname.split('/')[-2]]
        except:
            label = -1 # test has no label
        
        return img, label


def get_dataloaders(args):
    train_dataset = ImageDataset(args, training=True)
    test_dataset = ImageDataset(args, training=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, test_loader