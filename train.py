import os
import time
import random
import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split

from network.models import ConvNet
from data.loader import DatasetfromList, load_train_data_multi
from data.diffaug import DiffAugment
from utils.utils import calc_mean_accuracy, AverageMeter


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="batch train test")
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument(
        "--gpu_id", required=False, metavar="gpu_id", help="gpu id (0/1)"
    )
    parser.add_argument(
        "--output_name",
        default="train_adv_aug",
        type=str,
        help="prefix used to define output path",
    )
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--dataset", default="trainB", type=str)
    parser.add_argument(
        "--model", default=None, type=str, help="which model to use in diffaug test"
    )
    parser.add_argument("--resume", default="", type=str, help="which model to resume")
    parser.add_argument(
        "--epochs", default=1000, type=int, help="num of training epochs"
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument(
        "--augments", default=None, type=str, help="augmentation operations"
    )
    parser.add_argument("--adv_step", default=0.2, type=float, help="fgsm step size")
    parser.add_argument(
        "--eps", default=0.5, type=float, help="adversarial step: projection radias"
    )
    parser.add_argument("--n_repeats", default=3, type=int, help="adversarial repeat")
    # random augmentation
    parser.add_argument(
        "--random", action="store_true", default=False, help="abalation: random augs"
    )
    args = parser.parse_args()

    if args.gpu_id != None:
        assert torch.cuda.is_available()
        args.device = torch.device('cuda:{:s}'.format(args.gpu))
    else:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Use device: ",args.device)

    train_diffaug(args.output_name, args.resume, args)


def train_diffaug(output_name, resume, args):

    train_folder = args.dataset

    if not os.path.exists("exp"):
        os.mkdir("exp")
    output_root = "exp/" + output_name + "_" + args.dataset + "/"
    train_output_root = output_root + "train_results/"
    outputPath = train_output_root

    if not os.path.exists(output_root):
        os.mkdir(output_root)
    if not os.path.exists(train_output_root):
        os.mkdir(train_output_root)
    if not os.path.exists(outputPath + "checkpoint"):
        os.makedirs(outputPath + "checkpoint")

    imagePath = os.path.join(args.dataset_root, train_folder)
    labelName = "labels{:s}_train.csv".format(args.dataset[5:])
    labelPath = os.path.join(args.dataset_root, labelName)

    # Data
    xList, yList = load_train_data_multi([imagePath], [labelPath])
    xTrainList, xValidList = train_test_split(
        np.array(xList), test_size=0.1, random_state=42
    )
    yTrainList, yValidList = train_test_split(
        np.array(yList), test_size=0.1, random_state=42
    )
    print("\n######### Regression #########")
    print("Train data:", xTrainList.shape, yTrainList.shape)
    print("Valid data:", xValidList.shape, yValidList.shape)
    print("##############################\n")
    yuv_weight = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.14714119, -0.28886916, 0.43601035],
            [0.61497538, -0.51496512, -0.10001026],
        ]
    )

    train_dataset = DatasetfromList(
        xTrainList,
        yTrainList,
        transform=transforms.Compose(
            [
                transforms.Resize((66, 200)),
                transforms.ToTensor(),
            ]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True
    )
    val_dataset = DatasetfromList(
        xValidList,
        yValidList,
        transform=transforms.Compose(
            [
                transforms.Resize((66, 200)),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda rgb_img: torch.matmul(
                        rgb_img.permute(1, 2, 0), yuv_weight.transpose(0, 1)
                    ).permute(2, 0, 1)
                ),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True
    )

    # Model
    model = ConvNet().to(args.device)

    start_epoch = 0
    if resume is not "":
        start_epoch = int(resume)
        resume = outputPath + "checkpoint/checkpoint-" + resume + ".pth.tar"
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"]

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.net.parameters(), args.lr)

    train_log = open(outputPath + "train-log", "w")

    if args.seed is not None:
        set_seed(args.seed)

    for epoch in range(start_epoch, args.epochs):

        train(train_loader, model, loss_fn, optimizer, epoch, train_log, args)

        validate(val_loader, model, loss_fn, epoch, train_log, args)

        if (epoch + 1) % 1 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                outputPath + "checkpoint/checkpoint-" + str(epoch) + ".pth.tar",
            )


def train(train_loader, model, loss_fn, optimizer, epoch, train_log, args):
    losses = AverageMeter()
    meanacc = AverageMeter()
    aug_set = DiffAugment(args.eps)

    if args.random:
        uniform = torch.distributions.uniform.Uniform(-args.eps, args.eps)

    model.train()
    start_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        aug_set.resample()
        for aug in args.augments:
            # input: tensor in shape [N, C, H, W] pixel value range: (0.0 - 1.0)
            input = input.to(args.device, non_blocking=True)
            target = target.view(-1, 1).to(args.device, non_blocking=True)

            delta = Variable(torch.zeros([1])).to(args.device)

            if args.random:
                delta = uniform.sample()
                delta = delta.unsqueeze(0).to(args.device)
                if aug == "1" and delta < 0:
                    delta = -delta
            else:
                with torch.enable_grad():
                    for _iter in range(args.n_repeats):
                        delta.requires_grad_(True)
                        if aug in ["1", "2", "R", "G", "B", "H", "S", "V"]:
                            x, param_min = aug_set(input, aug, delta)
                        elif aug == "N":
                            break
                        else:
                            raise NotImplementedError

                        output = model(x)
                        loss = loss_fn(output, target)

                        grad = torch.autograd.grad(
                            loss,
                            [delta],
                            grad_outputs=None,
                            only_inputs=True,
                            allow_unused=True,
                        )[0]
                        delta.data += args.adv_step * torch.sign(grad.data)
                        delta = torch.clamp(delta, param_min, args.eps)
                        delta.detach_()

            x, _ = aug_set(input, aug, delta)
            output = model(x)
            loss = loss_fn(output, target)
            acc = calc_mean_accuracy(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), input.size(0))
            meanacc.update(acc)

    print(
        "Epoch [{}/{}] \t Time: {:.2f} \t augmentation: {} \t mloss: {:.4f} \t macc: {:.4f}".format(
            epoch, args.epochs, time.time() - start_time, aug, losses.avg, meanacc.avg
        )
    )
    train_log.write(
        "Epoch [{}/{}] \t Time: {:.2f} \t augmentation: {} \t mloss: {:.4f} \t macc: {:.4f} \n".format(
            epoch, args.epochs, time.time() - start_time, aug, losses.avg, meanacc.avg
        )
    )


def validate(val_loader, model, loss_fn, epoch, log, args):
    losses = AverageMeter()
    meanacc = AverageMeter()

    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.to(args.device, non_blocking=True)
        target = target.view(-1, 1).to(args.device, non_blocking=True)

        output = model(input)

        val_loss = loss_fn(output, target)
        val_acc = calc_mean_accuracy(output, target)

        losses.update(val_loss.item(), input.size(0))
        meanacc.update(val_acc)

    print(
        "Val [{}/{}] \t val_loss: {:.4f} \t val_acc: {:.4f}".format(
            epoch, args.epochs, losses.avg, meanacc.avg
        )
    )
    log.write(
        "Val [{}/{}] \t val_loss: {:.4f} \t val_acc: {:.4f} \n".format(
            epoch, args.epochs, losses.avg, meanacc.avg
        )
    )


if __name__ == "__main__":
    main()
