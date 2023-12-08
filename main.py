import argparse, torch, os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from src.dataset import get_dataloaders
from src.model import Resnet

myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


def parser():
    ap = argparse.ArgumentParser()
    
    ap.add_argument('-i', '--input_image_dir', type=str, 
                    help='Input image directory that contains train/test subfolders, and each of them contains classname subfolders')
    ap.add_argument('-o', '--output_model_dir', type=str, default='./models', 
                    help='Output model directory')
    ap.add_argument('-m', '--model_path', type=str, default='', 
                    help='Restore specific checkpoints for either retraining/testing')
    ap.add_argument('--train', action='store_true', 
                    help='Whether to train the model')
    ap.add_argument('--test', action='store_true', 
                    help='Whether to make prediction')
    ap.add_argument('--device', type=str, default='cuda', 
                    help='Device used to run the code')
    ap.add_argument('--log_path', type=str, default='./run_log.txt', 
                    help='Path to save running log')
    ap.add_argument('--save_every', type=int, default=5, 
                    help='Save interval for training')
    
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--num_epochs', type=int, default=100)
    
    return ap.parse_args()


def train(args, train_loader, test_loader):
    num_classes = len(os.listdir(os.path.join(args.input_image_dir, 'test')))
    model = Resnet(num_classes).to(args.device)
    start_epoch = 1
    
    if args.model_path:
        if args.device == 'cuda':
            model_dict = torch.load(args.model_path)
        else:
            model_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['model'])
        start_epoch = model_dict['epoch']
        print(f'Restore checkpoint from epoch #{start_epoch}')
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        model.train()

        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            imgs, labels = batch
            logits = model(imgs.to(args.device))
            loss = criterion(logits, labels.to(args.device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = (logits.argmax(dim=-1) == labels.to(args.device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)
            
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[ Train | {epoch:03d}/{args.num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        model.eval()
        valid_loss = []
        valid_accs = []
        
        for batch in tqdm(test_loader):
            imgs, labels = batch
            
            with torch.no_grad():
                logits = model(imgs.to(args.device))
            loss = criterion(logits, labels.to(args.device))
            
            acc = (logits.argmax(dim=-1) == labels.to(args.device)).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)
        
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        
        print(f"[ Valid | {epoch:03d}/{args.num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        with open(args.log_path, 'a') as f:
            print(f"[ Valid | {epoch:03d}/{args.num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}", file=f)

        if epoch % args.save_every == 0:
            print(f"Checkpoint saved at epoch #{epoch}")
            torch.save({'epoch': epoch, 'model': model.state_dict()}, os.path.join(args.output_model_dir, f'{epoch}.ckpt'))


def test(args, test_loader):
    num_classes = len(os.listdir(os.path.join(args.input_image_dir, 'test')))
    model = Resnet(num_classes).to(args.device)
    
    if args.model_path:
        if args.device == 'cuda':
            model_dict = torch.load(args.model_path)
        else:
            model_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict['model'])
    
    valid_accs = []
        
    for batch in tqdm(test_loader):
        imgs, labels = batch
        
        with torch.no_grad():
            logits = model(imgs.to(args.device))
        
        acc = (logits.argmax(dim=-1) == labels.to(args.device)).float().mean()
        valid_accs.append(acc)
    
    valid_acc = sum(valid_accs) / len(valid_accs)
    
    print(f'Testing accuracy: {valid_acc:.5f}')


if __name__ == '__main__':
    args = parser()
    
    train_loader, test_loader = get_dataloaders(args)
    
    if args.train:
        os.makedirs(args.output_model_dir, exist_ok=True)
        train(args, train_loader, test_loader)
    if args.test:
        assert args.model_path, 'Please specify checkpoint path for testing!'
        test(args, test_loader)