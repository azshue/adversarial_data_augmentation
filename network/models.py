import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


class ConvNet(nn.Module):
    # The same architecture as in Bojarski et al, https://arxiv.org/abs/1604.07316
	def __init__(self):
		super(ConvNet, self).__init__()
		self.output_dim = 1
		self.net = self.get_model()
		for m in self.net.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def get_model(self):
		model = nn.Sequential(
			nn.Conv2d(3, 24, (5, 5), stride=2),
			nn.ELU(inplace=True),
			nn.Conv2d(24, 36, (5, 5), stride=2),
			nn.ELU(inplace=True),
			nn.Conv2d(36, 48, (5, 5), stride=2),
			nn.ELU(inplace=True),
			nn.Conv2d(48, 64, (3, 3)),
			nn.ELU(inplace=True),
			nn.Conv2d(64, 64, (3, 3)),
			nn.ELU(inplace=True),
			nn.Flatten(),
			nn.Linear(1152, 100),
			nn.ELU(inplace=True),
			nn.Linear(100, 50),
			nn.ELU(inplace=True),
			nn.Linear(50, 10),
			nn.ELU(inplace=True),
			nn.Linear(10, 1),
		)
		return model
    
	def forward(self, input):
		return self.net(input)