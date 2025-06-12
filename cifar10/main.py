"""
implementation of resnet18 and mobilenet follows https://github.com/kuangliu/pytorch-cifar
"""

from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
import torchvision.models as torchmodels

import os
import argparse

from utils import progress_bar
from utils import get_config
import moe
import resnet, mobilenet
from PIL import Image
from utils import entropy
import numpy as np
import random
import supported
import wandb

torch.cuda.set_device(0)
torch.manual_seed(1)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

config = get_config()
EXPERT_NUM = config["experts"]
CLUSTER_NUM = config["clusters"]
strategy = config["strategy"]
PATIENCE = config["patience"]
MAX_EPOCH = config["max_epoch"]

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--model", choices=supported.models)
parser.add_argument(
    "--mixture", action="store_true", help="use MoE model instead of single model"
)
parser.add_argument("--resume", "-r", help="resume from checkpoint")
parser.add_argument(
    "--batch_size", type=int, default=128, help="input batch size for training"
)
parser.add_argument(
    "--do_early_stop", action = "store_true", help="input batch size for training"
)
parser.add_argument("--note", default=None, help="note for the model")
parser.add_argument("--wandb_id", default=None, help="id for the wandb run")

args = parser.parse_args()


device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
best_test_loss = np.inf
best_acc = 0  # best test accuracy
best_acc_list = []
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = args.batch_size
wandb_id = args.wandb_id
DO_EARLY_STOP = args.do_early_stop
checkpoint_name = f"ckpt_{'moe' if args.mixture else 'norm'}_{args.model}_batch_size_{batch_size}_{args.note}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
resume_checkpoint = args.resume


# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.CenterCrop(24),
        transforms.Resize(size=32),
        transforms.ToTensor(),
        # transforms.GaussianBlur(kernel_size=(3,7), sigma=(1.1,2.2)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_train_aug_1 = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            32, scale=(0.8, 1.2), ratio=(0.75, 1.33)
        ),  # Random zoom & aspect ratio
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),  # Slight chance of vertical flip
        transforms.RandomRotation(15),  # Small rotations
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random shift
        transforms.RandomPerspective(
            distortion_scale=0.2, p=0.3
        ),  # Simulate 3D distortion
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_train_aug_2 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(
            p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value="random"
        ),
    ]
)


transform_test = transforms.Compose(
    [
        transforms.CenterCrop(24),
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Create trainset
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)

trainset_aug_1 = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train_aug_1
)

trainset_aug_2 = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train_aug_2
)


# Create cluster and targets
# trainset.targets = torch.tensor(trainset.targets)
# trainset.cluster = trainset.targets
# trainset.targets = torch.zeros_like(trainset.targets)

trainset = torch.utils.data.ConcatDataset([trainset, trainset_aug_1, trainset_aug_2])
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g,
)


# Testset cluster and targets
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
# testset.targets = torch.tensor(testset.targets)
# testset.cluster = testset.targets
# testset.targets = torch.zeros_like(testset.targets)

# # Testset negative
# testset_flip = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                         download=True, transform=transform_rotate_test)
# testset_flip.targets = torch.tensor(testset_flip.targets)
# testset_flip.cluster = testset_flip.targets
# testset_flip.targets = torch.ones_like(testset_flip.targets)

# testset = torch.utils.data.ConcatDataset([testset,testset_flip])
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g,
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        for optim in optimizers:
            optim.zero_grad()
        if args.mixture:
            outputs, _, load_balance_loss, _ = net(inputs)
            clf_loss = criterion(outputs, targets)
            loss = clf_loss + 0.001*load_balance_loss
        
        else:
            if args.model == "resnet18":
                outputs, _ = net(inputs)
            else:
                outputs = net(inputs)
            loss = criterion(outputs, targets)
        loss.backward()

        for optim in optimizers:
            optim.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_acc = 100.*correct/total
    train_loss = train_loss/(batch_idx+1)
    return train_acc, train_loss
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            clusters = targets
            if args.mixture:
                outputs, select0, _, _ = net(inputs)
            else:
                if args.model == "resnet18":
                    outputs, _ = net(inputs)
                else:
                    outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Test loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Save checkpoint.
    test_acc = 100.*correct/total
    test_loss = test_loss/(batch_idx+1)
    if test_acc > best_acc:
        print('Saving..')
        state = {
            "net": net.state_dict(),
            "acc": test_acc,
            "epoch": epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = test_acc
    return test_acc, test_loss

if __name__ == "__main__":
    # for i in range(5):
    import wandb
    for i in range(1):
        print(f"Model: {args.model} | Mixture: {args.mixture}")
        print(f"Batch size: {batch_size}")
        print("==> Building model..")

        if args.model == "resnet18":
            if args.mixture:
                net = moe.NonlinearMixtureRes(EXPERT_NUM, strategy=strategy).to(device)
                # optimizer = moe.NormalizedGD(net.models.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
                optimizer = moe.NormalizedGD(
                    net.models.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
                )
                optimizer2 = optim.SGD(
                    net.router.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4
                )
                optimizers = [optimizer, optimizer2]
            else:
                net = resnet.ResNet18().to(device)
                optimizer = optim.SGD(
                    net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4
                )
                optimizers = [optimizer]

        elif args.model == "MobileNetV2":
            if args.mixture:
                net = moe.NonlinearMixtureMobile(EXPERT_NUM, strategy=strategy).to(
                    device
                )
                optimizer = moe.NormalizedGD(
                    net.models.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4
                )
                optimizer2 = optim.SGD(
                    net.router.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4
                )
                optimizers = [optimizer, optimizer2]
            else:
                net = mobilenet.MobileNetV2().to(device)
                optimizer = optim.SGD(
                    net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4
                )
                optimizers = [optimizer]

        if args.resume:
            # Load checkpoint.
            print("==> Resuming from checkpoint..")
            assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
            checkpoint = torch.load(f"./checkpoint/{resume_checkpoint}.pth")
            net.load_state_dict(checkpoint["net"])
            best_acc = checkpoint["acc"]
            start_epoch = checkpoint["epoch"]

        criterion = nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCH)

        ent_list, acc_list = [], []

        # Start a new wandb run to track this script.
        wandb.login()
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="retsam-deep-learning",
            # Set the wandb project where this run will be logged.
            project="machine-learning-data-mining",
            # id for the run
            id=wandb_id,
            # Resume a run that must use the same run ID.
            resume="allow",
            # Track hyperparameters and run metadata.
            config={
                "model": args.model,
                "is_mixture": args.mixture,
                "dataset": "CIFAR-10",
                "batch_size": batch_size,
                "epochs": MAX_EPOCH,
                "note": args.note,
            },
        )

        patience_count = 0
        for epoch in range(start_epoch, start_epoch + MAX_EPOCH):
            print(f"\nEpoch: {epoch + 1}/{MAX_EPOCH}")
            train_acc, train_loss = train(epoch)
            test_acc, test_loss = test(epoch)
            scheduler.step()

            run.log(
                {
                    "best_acc": best_acc,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                }
            )

            # Early stopping
            if DO_EARLY_STOP:
            # if epoch > PATIENCE:
                if patience_count >= PATIENCE:
                    print(f"Early stopping, stop at epoch <{epoch}>.")
                    break
                else:
                    if test_acc > best_acc:
                        patience_count = 0
                        best_acc = test_acc
                    else:
                        patience_count += 1

        run.finish()

        best_acc_list.append(best_acc)
        best_acc = 0

    print(
        f"Average accuracy: {np.mean(best_acc_list)} \t standard deviation: {np.std(best_acc_list)}"
    )
