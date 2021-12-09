import os
import glob
import argparse
import cv2


def crop_padding(dataset_path, folder):
	save_folder = dataset_path + folder + '_cropped'
	if not os.path.exists(save_folder):
		os.mkdir(save_folder)
	image_paths = glob.glob(dataset_path + folder + "/*.jpg")
	for image_path in image_paths:
		print(image_path)
		src_image = cv2.imread(image_path)
		h, w = src_image.shape[:2]

		cx = int(src_image.shape[1]/2)
		cy = int(src_image.shape[0]/2)
		pad_x = 0
		pad_y = 0
		for pix in range(src_image[cy, :, 0].shape[0]):
			if not src_image[cy, pix, 0] == 0:
				pad_x = pix
				break
		for pix in range(src_image[:, cx, 0].shape[0]):
			if not src_image[pix, cx, 0] == 0:
				pad_y = pix
				break
		print("pad x, y: ", pad_x, pad_y)

		new_image = src_image[pad_y:h-pad_y, pad_x:w-pad_x]
		new_image = cv2.resize(new_image, (w, h))
		new_image_path = image_path.replace(folder, folder+'_cropped')
		cv2.imwrite(new_image_path, new_image)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument(
        "--gpu_id", required=False, metavar="gpu_id", help="gpu id (0/1)"
    )
    parser.add_argument("--dataset", default="valB", type=str)
    args = parser.parse_args()

    for level in range(5):
        dir = "{:s}_distort_{:d}".format(args.dataset, level)
        crop_padding(args.dataset_root, dir)
