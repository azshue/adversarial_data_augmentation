import torch

def calc_mean_accuracy(y_pred, y_true):
	thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]
	total_acc = 0
	prediction_error = torch.abs(y_true-y_pred)
	# tf.print(prediction_error, tf.shape(prediction_error))

	for thresh_hold in thresh_holds:
		acc = torch.where(prediction_error < thresh_hold, torch.tensor(1.).cuda(), torch.tensor(0.).cuda())
		acc = torch.mean(acc)
		total_acc += acc

	ma = total_acc / len(thresh_holds)
	return ma

def calc_mean_accuracy_masked(y_pred, y_true, mask):
	thresh_holds = [0.1, 0.2, 0.5, 1, 2, 5]
	total_acc = 0
	prediction_error = torch.abs(y_true-y_pred)
	# tf.print(prediction_error, tf.shape(prediction_error))

	for thresh_hold in thresh_holds:
		acc = torch.where(prediction_error < thresh_hold, torch.tensor(1.).cuda(), torch.tensor(0.).cuda())
		acc = acc - (1 - mask)
		acc = torch.mean(acc) * 4
		total_acc += acc

	ma = total_acc / len(thresh_holds)
	return ma

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count