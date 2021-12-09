import os
import time
import csv
import torch
from torchvision import transforms

from network.models import ConvNet
from data.loader import DatasetfromList, load_train_data_multi
from utils.utils import calc_mean_accuracy, AverageMeter


def test_network(model, imagePath, labelPath, device):
    xList, yList = load_train_data_multi([imagePath], [labelPath])
    yuv_weight = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.14714119, -0.28886916, 0.43601035],
            [0.61497538, -0.51496512, -0.10001026],
        ]
    )
    test_dataset = DatasetfromList(
        xList,
        yList,
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
    test_iter = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True
    )
    meanacc = AverageMeter()

    for i, (input, target) in enumerate(test_iter):
        input = input.to(device, non_blocking=True)
        target = target.view(-1, 1).to(device, non_blocking=True)

        output = model(input)

        acc = calc_mean_accuracy(output, target)
        meanacc.update(acc)

    return meanacc.avg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="batch train test")
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument(
        "--gpu_id", required=False, metavar="gpu_id", help="specify the gpu to use"
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
    test_output_root = os.path.join(output_root, "test_results_{:s}".format(args.ckpt_epoch))
    if not os.path.exists(test_output_root):
        os.mkdir(test_output_root)
    train_output_root = os.path.join(output_root, "train_results")
    train_folder = args.dataset.replace("val", "train")

    outputPath = train_output_root

    modelPath = os.path.join(outputPath, "checkpoint/checkpoint-{:s}.pth.tar".format(args.ckpt_epoch))

    # get all testset with single corruption factors
    folders = [args.dataset]
    folders.extend(
        [f for f in os.listdir(args.dataset_root) if "{:s}_".format(args.dataset) in f]
    )
    folders = sorted(folders)

    val_folders = []
    # filter out unused folders
    for folder in folders:
        # use cropped distorted images
        if "distort" in folder:
            if "cropped" not in folder:
                continue
        # test a total of 25 combined images seperately in test_comb.py
        if "combined" in folder:
            continue
        if args.dataset == "valB":
            # the 'valB' test sets are named in different formats than others
            if "darker" in folder or "lighter" in folder:
                continue
        val_folders.append(folder)

    if args.dataset == "valB":
        for shade in ["lighter", "darker"]:
            for color in ["R", "G", "B", "H", "S", "V"]:
                folders = [
                    "{:s}{:s}/{:s}".format(shade, color, f)
                    for f in os.listdir(
                        os.path.join(args.dataset_root, "{:s}{:s}".format(shade, color))
                    )
                ]
                val_folders.extend(folders)
        val_folders.extend(
            ["noise/{:s}".format(f) for f in os.listdir(os.path.join(args.dataset_root, "noise"))]
        )
        val_folders.extend(
            ["blur/{:s}".format(f) for f in os.listdir(os.path.join(args.dataset_root, "blur"))]
        )
    val_folders = sorted(val_folders)
    val_types = [
        "B_darker",
        "B_lighter",
        "G_darker",
        "G_lighter",
        "H_darker",
        "H_lighter",
        "IMGC_fog",
        "IMGC_frost",
        "IMGC_jpeg_compression",
        "IMGC_motion_blur",
        "IMGC_pixelate",
        "IMGC_snow",
        "IMGC_zoom_blur",
        "R_darker",
        "R_lighter",
        "S_darker",
        "S_lighter",
        "V_darker",
        "V_lighter",
        "blur",
        "distort_cropped",
        "noise",
    ]
    print("val types: ", val_types)
    print("val folders: ", val_folders)
	# normalizers for computing mean Corrupted Error (mCE),
	# which are raw accuracies of the base model
    if args.dataset == "valB":
        normalizer = {
            "blur": 0.827552,
            "distort_cropped": 0.649412,
            "noise": 0.74875,
            "R_darker": 0.749182,
            "G_darker": 0.645504,
            "B_darker": 0.69151,
            "H_darker": 0.772466,
            "S_darker": 0.806022,
            "V_darker": 0.587342,
            "R_lighter": 0.668368,
            "G_lighter": 0.680266,
            "B_lighter": 0.765556,
            "H_lighter": 0.75302,
            "S_lighter": 0.70203,
            "V_lighter": 0.63646,
            "IMGC_zoom_blur": 0.804094,
            "IMGC_jpeg_compression": 0.887552,
            "IMGC_frost": 0.51507,
            "IMGC_motion_blur": 0.797968,
            "IMGC_snow": 0.514254,
            "IMGC_pixelate": 0.890748,
            "IMGC_fog": 0.427414,
        }
    elif args.dataset == "valHc":
        normalizer = {
            "blur": 0.708526,
            "distort_cropped": 0.596484,
            "noise": 0.653718,
            "R_darker": 0.707092,
            "G_darker": 0.576752,
            "B_darker": 0.685854,
            "H_darker": 0.691248,
            "S_darker": 0.693828,
            "V_darker": 0.565978,
            "R_lighter": 0.704996,
            "G_lighter": 0.447308,
            "B_lighter": 0.676036,
            "H_lighter": 0.698936,
            "S_lighter": 0.575588,
            "V_lighter": 0.444076,
            "IMGC_zoom_blur": 0.696604,
            "IMGC_jpeg_compression": 0.721722,
            "IMGC_frost": 0.351786,
            "IMGC_motion_blur": 0.686476,
            "IMGC_snow": 0.4335,
            "IMGC_pixelate": 0.723802,
            "IMGC_fog": 0.404308,
        }
    elif args.dataset == "valAds":
        normalizer = {
            "blur": 0.818334,
            "distort_cropped": 0.693952,
            "noise": 0.7276,
            "R_darker": 0.439194,
            "G_darker": 0.298426,
            "B_darker": 0.59035,
            "H_darker": 0.823682,
            "S_darker": 0.810796,
            "V_darker": 0.216226,
            "R_lighter": 0.52989,
            "G_lighter": 0.485252,
            "B_lighter": 0.74126,
            "H_lighter": 0.75057,
            "S_lighter": 0.339222,
            "V_lighter": 0.342754,
            "IMGC_zoom_blur": 0.783944,
            "IMGC_jpeg_compression": 0.939492,
            "IMGC_frost": 0.25573,
            "IMGC_motion_blur": 0.789208,
            "IMGC_snow": 0.268172,
            "IMGC_pixelate": 0.942866,
            "IMGC_fog": 0.13801,
        }

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
        test_output_root, "val_log_{:s}{:s}.log".format(os.path.basename(modelPath).replace(".", "-"), now02))
    output = open(outputPath, "w")
    val_logPath = os.path.join(
        test_output_root, "val_log_{:s}{:s}.csv".format(os.path.basename(modelPath).replace(".", "-"), now02))
    val_log = open(val_logPath, "wt", newline="")
    cw = csv.writer(val_log)
    cw.writerow([val_log])
    output.write("testing model: {}\n".format(modelPath))
    cw.writerow(["{:s} Set".format(args.dataset), "degree", "mean_accuracy"])

    labelName = "labels{:s}_val.csv".format(args.dataset[3:])
    labelPath = os.path.join(args.dataset_root, labelName)

    single_mCE = []
    unseen_mCE = []
    single_MA = []
    unseen_MA = []

    single_list = ["blur", "noise", "darker", "lighter"]

    cw.writerow([" ", "\t\t\t val type: clean \t\t\t", " "])
    output.write("\t\t\t val type: clean \t\t\t\n")
    val = args.dataset
    imagePath = os.path.join(args.dataset_root, val)
    MA = test_network(model, imagePath, labelPath, device)
    cw.writerow([val, "{:.3f}".format(100 * MA)])
    output.write("val folder: {}, \t mean accuracy: {:.3f}\n".format(val, 100 * MA))
    print("val folder: {}, \t mean accuracy: {:.3f}\n".format(val, 100 * MA))
    val_folders.remove(args.dataset)

    for t in val_types:
        cw.writerow([" ", "\t\t\t val type {:s}: \t\t\t".format(t), " "])
        output.write("\t\t\t val type: {:s} \t\t\t\n".format(t))
        degrees = [f for f in val_folders if t in f]
        if t == "blur":
            degrees = [d for d in degrees if "{}_blur_".format(args.dataset) in d]
        if t == "noise":
            degrees = [d for d in degrees if "{}_noise_".format(args.dataset) in d]
        type_err = 0
        assert len(degrees) > 0
        for degree in degrees:
            imagePath = os.path.join(args.dataset_root, degree)
            MA = test_network(model, imagePath, labelPath, device)
            type_err += 1.0 - MA
            if any(token in t for token in single_list):
                if "IMGC" not in t:
                    single_MA.append(MA)
                else:
                    unseen_MA.append(MA)
            else:
                unseen_MA.append(MA)
            cw.writerow([t, degree, "{:.3f}".format(100 * MA)])
            output.write(
                "val folder: {}, \t degree: {}, \t mean accuracy: {:.3f}\n".format(
                    t, degree, 100 * MA
                )
            )
            print(
                "val folder: {}, \t degree: {}, \t mean accuracy: {:.3f}".format(
                    t, degree, 100 * MA
                )
            )
        err_n = normalizer[t]
        type_err /= 1.0 * len(degrees) * (1.0 - err_n)
        if any(token in t for token in single_list):
            if "IMGC" not in t:
                single_mCE.append(type_err)
            else:
                unseen_mCE.append(type_err)
        else:
            unseen_mCE.append(type_err)
        cw.writerow([t, "mean Corrupted Error", "{:.3f}".format(100 * type_err)])
        output.write(
            "val type: {}, \t mean Corrupted Error: {:.3f}\n".format(t, 100 * type_err)
        )
        print("val type: {}, \t mean Corrupted Error: {:.3f}".format(t, 100 * type_err))

    cw.writerow(["Summary", " ", " "])
    cw.writerow(
        ["single_MA", "{:.3f}".format(100 * sum(single_MA) / len(single_MA)), " "]
    )
    cw.writerow(
        ["single_mCE", "{:.3f}".format(100 * sum(single_mCE) / len(single_mCE)), " "]
    )
    cw.writerow(
        ["unseen_MA", "{:.3f}".format(100 * sum(unseen_MA) / len(unseen_MA)), " "]
    )
    cw.writerow(
        ["unseen_mCE", "{:.3f}".format(100 * sum(unseen_mCE) / len(unseen_mCE)), " "]
    )

    val_log.close()
    output.close()
