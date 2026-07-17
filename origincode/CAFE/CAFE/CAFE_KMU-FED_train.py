import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import cv2
from ultralytics import YOLO
import random
import clip
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
clip_model, preprocess = clip.load("clip/ViT-B-32.pt", device=device)

# ===================== 超参数配置 =====================
num_classes = 7
input_size = (224, 224)
batch_size = 32
lr = 0.0002
num_epochs = 60
patience = 10
num_workers = 0

base_output = "KMU-FED/output_kmu_fed_clip/"
os.makedirs(base_output, exist_ok=True)

emotion_map = {
    "AN": 0, "DI": 1, "FE": 2, "HA": 3,
    "SA": 4, "SU": 5, "NE": 6
}
face_detector = YOLO("/home/chenruimin/chenruimin/mini_Xception-main/face_yolov8n.pt")

# ===================== 【复用】你原来的 KMU-FED 数据集 =====================
class KMU_FED(Dataset):
    def __init__(self, root_dir, input_size=(224,224), transform=None):
        self.root_dir = root_dir
        self.img_paths = []
        self.labels = []
        self.subject_ids = []

        for fname in os.listdir(root_dir):
            if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                parts = fname.split('_')
                subj_id = str(int(parts[0]))
                emo_code = parts[1]
                if emo_code in emotion_map:
                    self.img_paths.append(os.path.join(root_dir, fname))
                    self.labels.append(emotion_map[emo_code])
                    self.subject_ids.append(subj_id)

        self.transform = transform
        self.input_size = input_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        src = cv2.imread(self.img_paths[idx])
        img_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

        results = face_detector(src, conf=0.4)
        if len(results) > 0 and len(results[0].boxes) > 0:
            x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
            face = img_rgb[y1:y2, x1:x2]
        else:
            face = img_rgb

        if self.transform is not None:
            face = self.transform(face)

        label = self.labels[idx]
        return face, label

# ===================== 【直接移植】你的 CLIP+ResNet18 完整模型 =====================
class my_MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        input = input.transpose(3,1)


        input = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        input = input.transpose(3,1).contiguous()

        return input



class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
                
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            i = self.downsample(i)
                        
        x += i
        x = self.relu(x)
        
        return x

class ResNet(nn.Module):
    def __init__(self, block, n_blocks, channels, output_dim):
        super().__init__()
                
        
        self.in_channels = channels[0]
            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block=BasicBlock, n_blocks=[2,2,2,2], channels=[64, 128, 256, 512], stride = 1):
    
        layers = []
        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x, h

def Mask(nb_batch):
    bar = []
    for i in range(7):
        foo = [1] * 63 + [0] *  10
        if i == 6:
            foo = [1] * 64 + [0] *  10
        random.shuffle(foo)  #### generate mask
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch,512,1,1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar

def supervisor(x, targets, cnum):
    branch = x
    branch = branch.reshape(branch.size(0),branch.size(1), 1, 1)
    branch = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch)  
    branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
    loss_2 = 1.0 - 1.0*torch.mean(torch.sum(branch,2))/cnum # set margin = 3.0
    
    mask = Mask(x.size(0))
    branch_1 = x.reshape(x.size(0),x.size(1), 1, 1) * mask 
    branch_1 = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch_1)  
    branch_1 = branch_1.view(branch_1.size(0), -1)
    loss_1 = nn.CrossEntropyLoss()(branch_1, targets)

    return [loss_1, loss_2] 

class Model(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0):
        super(Model, self).__init__()
        
        res18 = ResNet(block = BasicBlock, n_blocks = [2,2,2,2], channels = [64, 128, 256, 512], output_dim=1000)
        msceleb_model = torch.load('clip/resnet18_msceleb.pth')
        state_dict = msceleb_model['state_dict']
        res18.load_state_dict(state_dict, strict=False)
        
        self.drop_rate = drop_rate
        self.features = nn.Sequential(*list(res18.children())[:-2])
        self.features2 = nn.Sequential(*list(res18.children())[-2:-1])
        
        fc_in_dim = list(res18.children())[-1].in_features  # original fc layer's in dimention 512
        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7
        
        self.parm={}
        for name,parameters in self.fc.named_parameters():
            print(name,':',parameters.size())
            self.parm[name]=parameters
        
    def forward(self, x, targets=None, phase='train'):
        with torch.no_grad():
            image_features = clip_model.encode_image(x)
            
        x = self.features(x)
        feat = x
        
        x = self.features2(x)
        x = x.view(x.size(0), -1)    
        ################### sigmoid mask (important)
        if phase=='train':
            MC_loss = supervisor(image_features * torch.sigmoid(x), targets, cnum=73)

        x = image_features * torch.sigmoid(x)
        out = self.fc(x)
        
        if phase=='train':
            return out, MC_loss
        else:
            return out

# ===================== 训练一折 =====================
def train_one_fold(model, train_loader, val_loader, fold_idx):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    best_val_acc = 0.0
    fold_dir = os.path.join(base_output, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs, mc_loss = model(imgs, labels, phase='train')

            loss_ce = F.cross_entropy(outputs, labels)
            loss = loss_ce + 5 * mc_loss[1] + 1.5 * mc_loss[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs, phase='test')
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(fold_dir, "best.pth"))

        print(f"Fold{fold_idx:2d} Epoch{epoch+1:2d} | Train {train_acc:.4f} | Val {val_acc:.4f}")

    print(f"✅ Fold{fold_idx} Best: {best_val_acc:.4f}")
    return best_val_acc

# ===================== 10折按受试者划分 =====================
def run_10fold():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.02, 0.25)) ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    dataset = KMU_FED("/data/chenruimin/KMU-FED", transform=val_transform)
    unique_subjects = np.array(sorted(set(dataset.subject_ids), key=int))
    print("总人数:", len(unique_subjects))

    skf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(unique_subjects)):
        print(f"\n========== Fold {fold+1} ==========")
        tr_persons = unique_subjects[tr_idx]
        va_persons = unique_subjects[va_idx]
        tr_set = set(tr_persons)

        tr_indices = [i for i,p in enumerate(dataset.subject_ids) if p in tr_set]
        va_indices = [i for i,p in enumerate(dataset.subject_ids) if p not in tr_set]

        train_dataset = KMU_FED("/data/chenruimin/KMU-FED", transform=train_transform)
        val_dataset = dataset

        train_loader = DataLoader(Subset(train_dataset, tr_indices), batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(Subset(val_dataset, va_indices), batch_size, shuffle=False, num_workers=num_workers)

        model = Model(num_classes=7).to(device)
        best = train_one_fold(model, train_loader, val_loader, fold+1)
        fold_accs.append(best)

    print("\n" + "="*60)
    for i, acc in enumerate(fold_accs):
        print(f"Fold {i+1}: {acc:.4f}")
    print(f"\n平均精度: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")

if __name__ == "__main__":
    run_10fold()