from collections import OrderedDict, defaultdict
from tqdm.auto import tqdm

from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *

from utils import *


class VGG(nn.Module):
    ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    def __init__(self) -> None:
        super().__init__()

        layers = []
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != 'M':
                # conv-bn-relu
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                # maxpool
                add("pool", nn.MaxPool2d(2))

        add("avgpool", nn.AvgPool2d(2))
        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
        x = self.backbone(x)

        # avgpool: [N, 512, 2, 2] => [N, 512]
        # x = x.mean([2, 3])
        x = x.view(x.shape[0], -1)

        # classifier: [N, 512] => [N, 10]
        x = self.classifier(x)
        return x

def train(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        scaler,
        use_amp,
        callbacks=None
) -> None:
    model.train()

    for inputs, targets in tqdm(dataloader, desc='train', leave=False):
        # Move the data from CPU to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Reset the gradients (from the last iteration)
        optimizer.zero_grad()

        # Forward inference
        with torch.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()  # Backward propagation
            scaler.step(optimizer)  # Update optimizer
            scaler.update()
        else:
            loss.backward()  # Backward propagation
            optimizer.step()  # Update optimizer

        if callbacks is not None:
            for callback in callbacks:
                callback()

@torch.inference_mode()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        extra_preprocess=None
) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
        # Move the data from CPU to GPU
        inputs = inputs.cuda()
        if extra_preprocess is not None:
            for preprocess in extra_preprocess:
                inputs = preprocess(inputs)

        targets = targets.cuda()

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()

def get_model_flops(model, inputs):
    num_macs = profile_macs(model, inputs)
    return num_macs

def get_model_size(model: nn.Module, data_width=32):
    """
    calculate the model size in bits
    :param data_width: #bits per element
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

genDir('checkpoints')
set_random_seeds()
device_check()

if __name__ == "__main__":

    train_flag = False  # evaluate
    # train_flag = True # train

    model_name = 'vgg'
    model = VGG().cuda()
    checkpoint_path = f"./checkpoints/best_{model_name}.pth.tar"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"=> loading checkpoint '{checkpoint_path}'")
        model.load_state_dict(checkpoint)
    else:
        train_flag = True

    recover_model = lambda: model.load_state_dict(checkpoint)

    transforms = {
        "train": Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
        "test": ToTensor(),
    }
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(
            root="data/cifar10",
            train=(split == "train"),
            download=True,
            transform=transforms[split],
        )
    dataloader = {}
    for split in ['train', 'test']:
        dataloader[split] = DataLoader(
            dataset[split],
            batch_size=512,
            shuffle=(split == 'train'),
            num_workers=0,
            pin_memory=True,
        )

    fp32_model_accuracy = evaluate(model, dataloader['test'])
    fp32_model_size = get_model_size(model)
    print(f"fp32 model has accuracy={fp32_model_accuracy:.2f}%")
    print(f"fp32 model has size={fp32_model_size / MiB:.2f} MiB")

    if train_flag:
        use_amp = True
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        num_epochs = 100
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0
        epoch = num_epochs
        while epoch > 0:
            train(model, dataloader['train'], criterion, optimizer, scheduler, scaler, use_amp)
            model_acc = evaluate(model, dataloader['test'])
            scheduler.step()  # Update LR scheduler

            torch.save(model.state_dict(), f"./checkpoints/last_{model_name}.pth.tar")

            if model_acc > best_acc:
                best_acc = model_acc
                torch.save(model.state_dict(), f"./checkpoints/best_{model_name}.pth.tar")

            print(f'        Epoch {num_epochs - epoch} Acc {model_acc:.2f}% / Best Acc : {best_acc:.2f}%')
            epoch -= 1
