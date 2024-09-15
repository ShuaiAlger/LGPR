

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class BasicLayer(nn.Module):
	"""
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False),
									  nn.ReLU(inplace = True),
									)

	def forward(self, x):
	  return self.layer(x)

class XFeatModel(nn.Module):
	"""
	   Implementation of architecture described in 
	   "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	"""

	def __init__(self):
		super().__init__()
		self.norm = nn.InstanceNorm2d(1)


		########### ⬇️ CNN Backbone & Heads ⬇️ ###########

		self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4),
			  						 nn.Conv2d (1, 24, 1, stride = 1, padding=0) )

		self.block1 = nn.Sequential(
										BasicLayer( 1,  4, stride=1),
										BasicLayer( 4,  8, stride=2),
										BasicLayer( 8,  8, stride=1),
										BasicLayer( 8, 24, stride=2),
									)

		self.block2 = nn.Sequential(
										BasicLayer(24, 24, stride=1),
										BasicLayer(24, 24, stride=1),
									 )

		self.block3 = nn.Sequential(
										BasicLayer(24, 64, stride=2),
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, 1, padding=0),
									 )
		self.block4 = nn.Sequential(
										BasicLayer(64, 64, stride=2),
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, stride=1),
									 )

		self.block5 = nn.Sequential(
										BasicLayer( 64, 128, stride=2),
										BasicLayer(128, 128, stride=1),
										BasicLayer(128, 128, stride=1),
										BasicLayer(128,  64, 1, padding=0),
									 )

		self.block_fusion =  nn.Sequential(
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, stride=1),
										nn.Conv2d (64, 64, 1, padding=0)
									 )

		self.heatmap_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 1, 1),
										nn.Sigmoid()
									)


		self.keypoint_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 65, 1),
									)


  		########### ⬇️ Fine Matcher MLP ⬇️ ###########

		self.fine_matcher =  nn.Sequential(
											nn.Linear(128, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 64),
										)
	def _unfold2d(self, x, ws = 2):
		"""
			Unfolds tensor in 2D with desired ws (window size) and concat the channels
		"""
		B, C, H, W = x.shape
		x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
			.reshape(B, C, H//ws, W//ws, ws**2)
		return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)

	def forward(self, x):
		"""
			input:
				x -> torch.Tensor(B, C, H, W) grayscale or rgb images
			return:
				feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
				keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
				heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

		"""
		#dont backprop through normalization
		with torch.no_grad():
			x = x.mean(dim=1, keepdim = True)
			x = self.norm(x)

		#main backbone
		x1 = self.block1(x)
		x2 = self.block2(x1 + self.skip1(x))
		x3 = self.block3(x2)
		x4 = self.block4(x3)
		x5 = self.block5(x4)

		#pyramid fusion
		x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
		x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
		feats = self.block_fusion( x3 + x4 + x5 )

		#heads
		heatmap = self.heatmap_head(feats) # Reliability map
		keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits

		return feats, keypoints, heatmap





class XFeatNetBuilder(nn.Module):
	AVAILABLE_MODELS = {
	"xfeat": XFeatModel
	}
	def __init__(
		self,
		backbone_name="xfeat",
		pretrained=True,
		crop_last_block=True,
		num_unfrozen_blocks=1,
		):
		super().__init__()
		self.backbone_name = backbone_name
		self.pretrained = pretrained
		self.num_unfrozen_blocks = num_unfrozen_blocks
		self.crop_last_block = crop_last_block

		self.xfeatnet = self.AVAILABLE_MODELS[self.backbone_name]()

		if self.pretrained:
			weights_path = "/media/shuai/Correspondence/explore/XFeat/weights/xfeat.pt"
			# weights_path = "/media/shuai/Correspondence/explore/XFeat_Train/ckpts/xfeat_synthetic_9000.pt"
			if torch.cuda.is_available():
				# Train on GPU, deploy on GPU.
				self.xfeatnet.load_state_dict(torch.load(weights_path))
				self.xfeatnet = self.xfeatnet.cuda()
			else:
				# Train on GPU, deploy on CPU.
				self.xfeatnet.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))

		self.xfeatnet.block1.requires_grad_(False)
		self.xfeatnet.skip1.requires_grad_(False)
		self.xfeatnet.block2.requires_grad_(False)

		self.xfeatnet.block3.requires_grad_(False)
		self.xfeatnet.block4.requires_grad_(False)
		self.xfeatnet.block5.requires_grad_(False)

		self.xfeatnet.block_fusion.requires_grad_(False)

		self.out_channels = 64 * 3
		self.batch_idx = 0


	def forward(self, x):
		x1 = self.xfeatnet.block1(x)
		x2 = self.xfeatnet.block2(x1 + self.xfeatnet.skip1(x))
		x3 = self.xfeatnet.block3(x2)
		x4 = self.xfeatnet.block4(x3)
		x5 = self.xfeatnet.block5(x4)

		#pyramid fusion
		# sparse
		if 0:
			x3 = F.interpolate(x3, (x5.shape[-2], x5.shape[-1]), mode='bilinear')
			x4 = F.interpolate(x4, (x5.shape[-2], x5.shape[-1]), mode='bilinear')
		else:
			x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
			x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
		# feats = self.xfeatnet.block_fusion( x3 + x4 + x5 )



		# feats = self.xfeatnet.block_fusion( x3 + x4 + x5 )

		#heads
		# heatmap = self.xfeatnet.heatmap_head(feats) # Reliability map
		# keypoints = self.xfeatnet.keypoint_head(self.xfeatnet._unfold2d(x, ws=8)) #Keypoint map logits

		# if display:
		# input_show = x.detach().cpu()[0][0].numpy()
		# feats_show = feats.detach().cpu()[0][0].numpy()
		# heatmap_show = heatmap.detach().cpu()[0][0].numpy()
		# keypoints_show = keypoints.detach().cpu()[0][0].numpy()

		# import numpy as np
		# import cv2

		# feats_show = np.uint8(255*(feats_show - feats_show.min())/(feats_show.max() - feats_show.min()))
		# heatmap_show = np.uint8(255*(heatmap_show - heatmap_show.min())/(heatmap_show.max() - heatmap_show.min()))
		# keypoints_show = np.uint8(255*(keypoints_show - keypoints_show.min())/(keypoints_show.max() - keypoints_show.min()))

		# feats_show = cv2.applyColorMap(feats_show, cv2.COLORMAP_JET)
		# heatmap_show = cv2.applyColorMap(heatmap_show, cv2.COLORMAP_JET)
		# keypoints_show = cv2.applyColorMap(keypoints_show, cv2.COLORMAP_JET)

		# # index_save = np.uint32(10000 * np.random.random())
		# index_save = self.batch_idx

		# cv2.imwrite("/media/shuai/Correspondence/explore/OpenJKL/display/"+str(index_save)+"input_show.png", cv2.normalize(input_show, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
		# cv2.imwrite("/media/shuai/Correspondence/explore/OpenJKL/display/"+str(index_save)+"feats_show.png", feats_show)
		# cv2.imwrite("/media/shuai/Correspondence/explore/OpenJKL/display/"+str(index_save)+"heatmap_show.png", heatmap_show)
		# cv2.imwrite("/media/shuai/Correspondence/explore/OpenJKL/display/"+str(index_save)+"keypoints_show.png", keypoints_show)

		self.batch_idx = self.batch_idx + 1

		feats = torch.concat([x3, x4, x5], dim=1)

		return feats



