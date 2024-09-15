import torch
import torch.nn as nn



class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        return semi, desc
  

class SuperPointNetBuilder(nn.Module):
    AVAILABLE_MODELS = {
        "superpoint": SuperPointNet
    }
    def __init__(
        self,
        backbone_name="superpoint",
        pretrained=True,
        crop_last_block=True,
        num_unfrozen_blocks=1,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.crop_last_block = crop_last_block
       
        self.superpointnet = self.AVAILABLE_MODELS[self.backbone_name]()

        # weights_path = "/media/shuai/Correspondence/AllMatchingToolbox/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth"
        # if torch.cuda.is_available():
        #   # Train on GPU, deploy on GPU.
        #   self.superpointnet.load_state_dict(torch.load(weights_path))
        #   self.superpointnet = self.superpointnet.cuda()
        # else:
        #   # Train on GPU, deploy on CPU.
        #   self.superpointnet.load_state_dict(torch.load(weights_path,
        #                           map_location=lambda storage, loc: storage))

        # print(">>>   [self.superpointnet.conv1a.parameters] : ", self.superpointnet.conv1a.weight)
        # ckpt2 = "/media/shuai/Correspondence/explore/OpenVPRLab/logs/superpoint/BoQ/version_12/checkpoints/epoch(37)_step(5396)_R1[0.7608]_R5[0.8297].ckpt"
        # print(torch.load(ckpt2))

        # self.superpointnet.conv1a.requires_grad_(False)
        # self.superpointnet.conv1b.requires_grad_(False)
        # self.superpointnet.conv2a.requires_grad_(False)
        # self.superpointnet.conv2b.requires_grad_(False)
        # self.superpointnet.conv3a.requires_grad_(False)
        # self.superpointnet.conv3b.requires_grad_(False)
        # self.superpointnet.conv4a.requires_grad_(False)
        # self.superpointnet.conv4b.requires_grad_(False)


        self.out_channels = 128

    def forward(self, x):
        if 0:
            with torch.no_grad():
                x = self.superpointnet.relu(self.superpointnet.conv1a(x))
                x1 = self.superpointnet.relu(self.superpointnet.conv1b(x))
                x = self.superpointnet.pool(x1)
                x = self.superpointnet.relu(self.superpointnet.conv2a(x))
                x2 = self.superpointnet.relu(self.superpointnet.conv2b(x))
                x = self.superpointnet.pool(x2)
                x = self.superpointnet.relu(self.superpointnet.conv3a(x))
                x3 = self.superpointnet.relu(self.superpointnet.conv3b(x))
                x = self.superpointnet.pool(x3)
                x = self.superpointnet.relu(self.superpointnet.conv4a(x))
                x4 = self.superpointnet.relu(self.superpointnet.conv4b(x))
        else:
            x = self.superpointnet.relu(self.superpointnet.conv1a(x))
            x1 = self.superpointnet.relu(self.superpointnet.conv1b(x))
            x = self.superpointnet.pool(x1)
            x = self.superpointnet.relu(self.superpointnet.conv2a(x))
            x2 = self.superpointnet.relu(self.superpointnet.conv2b(x))
            x = self.superpointnet.pool(x2)
            x = self.superpointnet.relu(self.superpointnet.conv3a(x))
            x3 = self.superpointnet.relu(self.superpointnet.conv3b(x))
            x = self.superpointnet.pool(x3)
            x = self.superpointnet.relu(self.superpointnet.conv4a(x))
            x4 = self.superpointnet.relu(self.superpointnet.conv4b(x))
        return x




