import os
import time
import csv

import torch
from torchvision import transforms
import numpy as np

from network.models import ConvNet
from data.loader import DatasetfromList, load_train_data_multi
from data.diffaug import DiffAugment
from utils.utils import calc_mean_accuracy, AverageMeter


def test_network(model, imagePath, labelPath, aug_ids, delta, idx, device):
    xList, yList = load_train_data_multi([imagePath], [labelPath])
    test_dataset = DatasetfromList(
        xList,
        yList,
        transform=transforms.Compose(
            [
                transforms.Resize((66, 200)),
                transforms.ToTensor(),
            ]
        ),
    )
    delta = torch.tensor(delta).float().to(device)
    test_iter = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True
    )
    meanacc = AverageMeter()
    aug_set = DiffAugment(0.25)  # param here doesn't matter

    for i, (input, target) in enumerate(test_iter):
        input = input.to(device, non_blocking=True)
        target = target.view(-1, 1).to(device, non_blocking=True)
        x = aug_set(input, aug_ids, delta)

        output = model(x)
        acc = calc_mean_accuracy(output, target)
        meanacc.update(acc)

    return meanacc.avg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="batch train test")
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument(
        "--gpu_id", required=False, metavar="gpu_id", help="gpu id (0/1)"
    )
    parser.add_argument(
        "--exp_name",
        default="diffaug_results",
        type=str,
        help="name of the experiment (for locating the checkpoint dir)",
    )
    parser.add_argument("--dataset", default="valB", type=str)
    parser.add_argument(
        "--ckpt_epoch", default=None, type=str, help="which model to use in diffaug test"
    )
    args = parser.parse_args()

    if args.gpu_id != None:
        assert torch.cuda.is_available()
        device = torch.device('cuda:{:s}'.format(args.gpu))
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Use device: ", device)

    output_root = os.path.join("exp", "{:s}_{:s}".format(args.exp_name, args.dataset.replace("val", "train")))
    test_output_root = output_root + "test_results_" + args.ckpt_epoch + "/"
    if not os.path.exists(test_output_root):
        os.mkdir(test_output_root)
    train_output_root = os.path.join(output_root, "train_results")
    train_folder = args.dataset.replace("val", "train")

    outputPath = train_output_root

    modelPath = os.path.join(outputPath, "checkpoint/checkpoint-{:s}.pth.tar".format(args.ckpt_epoch))

    val_folders = [args.dataset]

    checkpoint = torch.load(modelPath)
    model = ConvNet().to(device)
    model.eval()
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        model.load_state_dict(
            {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        )

    now = int(round(time.time() * 1000))
    now02 = time.strftime("%Y%m%d-%H-%M-%S", time.localtime(now / 1000))
    outputPath = os.path.join(
        test_output_root, "val_comb_log_{:s}{:s}.log".format(os.path.basename(modelPath).replace(".", "-"), now02))
    output = open(outputPath, "w")
    param_log = open("./comb_param.txt", "w")
    val_logPath = os.path.join(
        test_output_root, "val_comb_log_{:s}{:s}.csv".format(os.path.basename(modelPath).replace(".", "-"), now02))
    val_log = open(val_logPath, "wt", newline="")
    cw = csv.writer(val_log)
    cw.writerow([val_log])
    output.write("testing model: {}\n".format(modelPath))
    cw.writerow(["valB Set", "degree", "mean_accuracy"])

    labelName = "labels{:s}_val.csv".format(args.dataset[3:])
    labelPath = os.path.join(args.dataset_root, labelName)
    # fixed random seed for generating the same combined factors on every run
    rng = np.random.default_rng(seed=0)
    aug_ids = np.arange(8)

    comb_MA = []
    for val in val_folders:
        cw.writerow([" ", "\t\t\t valB_{:s} \t\t\t".format(val), " "])
        output.write("\t\t\t valB_{:s} \t\t\t\n".format(val))
        # test a total of 25 combinations
        for i in range(25):
            imagePath = os.path.join(args.dataset_root, val)

            rng.shuffle(aug_ids)
            print("order {}: {}".format(i, aug_ids))
            delta = rng.normal(0, 0.33, size=8)
            print("delta {}: {}".format(i, delta))
            param_log.write("order {}: {}\n".format(i, aug_ids))
            param_log.write("delta {}: {}\n".format(i, delta))

            MA = test_network(
                model, imagePath, labelPath, aug_ids, delta, i, device
            )
            cw.writerow([val, "{:.3f}".format(100 * MA)])
            output.write("Combined: {}, \t mean accuracy: {:.3f}\n".format(i, 100 * MA))
            print("Combined: {}, \t mean accuracy: {:.3f}".format(i, 100 * MA))
            comb_MA.append(MA)

    MA = 100 * sum(comb_MA) / len(comb_MA)
    if args.dataset == "valB":
        err_n = 0.51440
    elif args.dataset == "valAds":
        err_n = 0.25231
    elif args.dataset == "valHc":
        err_n = 0.59926
    mCE = 100 * (1.0 - sum(comb_MA) / len(comb_MA)) / (1.0 - err_n)
    print("multi_MA: {:.3f}".format(MA))
    print("multi_mCE: {:.3f}".format(mCE))

    cw.writerow(["Summary", " ", " "])
    cw.writerow(["multi_MA", "{:.3f}".format(MA), " "])
    cw.writerow(["multi_mCE", "{:.3f}".format(mCE), " "])

    val_log.close()
    output.close()
