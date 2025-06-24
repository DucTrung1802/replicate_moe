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
import wandb
import json
from load_config import load_config


torch.cuda.set_device(0)
torch.manual_seed(1)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


# LOAD CONFIG =================================================================
FINAL_CONFIG = load_config("config.json")
# =============================================================================


# GLOBAL VARIABLES ============================================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
best_test_loss = np.inf
best_acc = 0  # best test accuracy
best_acc_list = []
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
checkpoint_name = f"ckpt_{'moe' if FINAL_CONFIG['mixture'] else 'norm'}_{FINAL_CONFIG['model']}_batch_size_{FINAL_CONFIG['batch_size']}_{FINAL_CONFIG['note']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
batch_size = FINAL_CONFIG["batch_size"]

# =============================================================================


# region DATA LOADING ==========================================================
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

# transform_rotate_train = transforms.Compose([
#     torchvision.transforms.RandomRotation((30,30)),
#     transforms.CenterCrop(24),
#     transforms.Resize(size=32),
#     transforms.ToTensor(),
#     transforms.GaussianBlur(kernel_size=(3,7), sigma=(1.1,2.2)),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_rotate_test = transforms.Compose([
#     torchvision.transforms.RandomRotation((30,30)),
#     transforms.CenterCrop(24),
#     transforms.Resize(size=32),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

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

# # trainset negative examples
# trainset_flip = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform_rotate_train)
# # Cluster and targets
# trainset_flip.targets = torch.tensor(trainset_flip.targets)
# trainset_flip.cluster = trainset_flip.targets
# trainset_flip.targets = torch.ones_like(trainset_flip.targets)

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


# endregion ===================================================================


# region TRAIN FUCNTION =======================================================
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        for optim in optimizers:
            optim.zero_grad()
        if FINAL_CONFIG["mixture"]:
            outputs, _, load_balance_loss, _ = net(inputs)
            clf_loss = criterion(outputs, targets)
            loss = clf_loss + 0.001 * load_balance_loss

        else:
            if FINAL_CONFIG["model"] == "resnet18":
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

        progress_bar(
            batch_idx,
            len(trainloader),
            "Train loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

    train_acc = 100.0 * correct / total
    train_loss = train_loss / (batch_idx + 1)

    return train_acc, train_loss


# endregion ===================================================================


# region TEST FUCNTION ========================================================
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
            if FINAL_CONFIG["mixture"]:
                outputs, select0, _, _ = net(inputs)
            else:
                if FINAL_CONFIG["model"] == "resnet18":
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
    test_acc = 100.0 * correct / total
    test_loss = test_loss / (batch_idx + 1)

    return test_acc, test_loss


# endregion ===================================================================


# region MAIN =================================================================
if __name__ == "__main__":
    print("Loading config...")
    print(json.dumps(FINAL_CONFIG, indent=2))

    for i in range(1):
        print("==> Building model..")

        model = FINAL_CONFIG["model"]
        mixture = FINAL_CONFIG["mixture"]
        expert_num = FINAL_CONFIG["expert_num"]
        strategy = FINAL_CONFIG["strategy"]
        resume = FINAL_CONFIG["resume"]
        max_epoch = FINAL_CONFIG["max_epoch"]
        note = FINAL_CONFIG["note"]
        early_stop = FINAL_CONFIG["early_stop"]
        learning_rate = FINAL_CONFIG["learning_rate"]
        weight_decay = FINAL_CONFIG["weight_decay"]
        momentum = FINAL_CONFIG["momentum"]
        wandb_name = FINAL_CONFIG["wandb_name"]
        wandb_resume_id = FINAL_CONFIG["wandb_resume_id"]
        checkpoint_resume_name = FINAL_CONFIG["resume_from_file"]

        if model == "resnet18":
            if mixture:
                net = moe.NonlinearMixtureRes(
                    expert_num=expert_num, strategy=strategy
                ).to(device)
                # optimizer = moe.NormalizedGD(net.models.parameters(), lr=1e-2, momentum=momentum, weight_decay=weight_decay)
                optimizer = moe.NormalizedGD(
                    net.models.parameters(),
                    lr=learning_rate,  # Default: 0.1
                    momentum=momentum,
                    weight_decay=weight_decay,
                )
                optimizer2 = optim.NAdam(
                    net.router.parameters(),
                    lr=learning_rate,  # Default: 1e-4
                    weight_decay=weight_decay,
                )
                optimizers = [optimizer, optimizer2]
            else:
                net = resnet.ResNet18().to(device)
                optimizer = optim.NAdam(
                    net.parameters(),
                    lr=learning_rate,  # Default: 1e-2
                    weight_decay=weight_decay,
                )
                optimizers = [optimizer]

        elif model == "MobileNetV2":
            if mixture:
                net = moe.NonlinearMixtureMobile(
                    expert_num=expert_num, strategy=strategy
                ).to(device)
                optimizer = moe.NormalizedGD(
                    net.models.parameters(),
                    lr=learning_rate,  # Default: 1e-2
                    momentum=momentum,
                    weight_decay=weight_decay,
                )
                optimizer2 = optim.NAdam(
                    net.router.parameters(),
                    lr=learning_rate,  # Default: 1e-4
                    weight_decay=weight_decay,
                )
                optimizers = [optimizer, optimizer2]
            else:
                net = mobilenet.MobileNetV2().to(device)
                optimizer = optim.NAdam(
                    net.parameters(),
                    lr=learning_rate,  # Default: 1e-2
                    weight_decay=weight_decay,
                )
                optimizers = [optimizer]

        if resume:
            # Load checkpoint.
            print("==> Resuming from checkpoint..")
            assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
            checkpoint = torch.load(f"./checkpoint/{checkpoint_resume_name}.pth")
            net.load_state_dict(checkpoint["net"])
            best_acc = checkpoint["acc"]
            start_epoch = checkpoint["epoch"]

        criterion = nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epoch
        )
        FINAL_CONFIG["lr_scheduler"] = scheduler.__class__.__name__

        ent_list, acc_list = [], []

        # Start a new wandb run to track this script.
        if FINAL_CONFIG["wandb_upload"]:
            wandb.login()
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="retsam-deep-learning",
                # Set the wandb project where this run will be logged.
                project="machine-learning-data-mining",
                # id for the run
                id=wandb_resume_id,
                # Resume a run that must use the same run ID.
                resume="allow",
                # Assign a name to the run
                name=wandb_name,
                # Track hyperparameters and run metadata.
                config=FINAL_CONFIG,
            )

        patience_count = 0
        for epoch in range(start_epoch, start_epoch + max_epoch):
            print(f"\nEpoch: {epoch + 1}/{max_epoch}")
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
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

            # Save loss values
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")

            # Early stopping and model checkpointing
            if test_acc > best_acc:
                print("Saving..")
                state = {
                    "net": net.state_dict(),
                    "acc": test_acc,
                    "epoch": epoch,
                }
                os.makedirs("checkpoint", exist_ok=True)
                torch.save(state, f"./checkpoint/{checkpoint_name}.pth")
                best_acc = test_acc
                patience_count = 0  # reset on improvement
            else:
                if early_stop:
                    patience_count += 1
                    if patience_count > FINAL_CONFIG["patience"]:
                        print(
                            f"Early stopping at epoch {epoch} with best_acc {best_acc:.4f}"
                        )
                        break

        if FINAL_CONFIG["wandb_upload"]:
            run.finish()

        best_acc_list.append(best_acc)
        best_acc = 0

    print(
        f"Average accuracy: {np.mean(best_acc_list)} \t standard deviation: {np.std(best_acc_list)}"
    )
