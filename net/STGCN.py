import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.tgcn import ConvTemporalGraphical
from net.graph import Graph

class MutltiViewModel(nn.Module):

    def __init__(self, camera, in_channels, num_class, graph_args,
                edge_importance_weighting, attention, upper_bound = 4, **kwargs):
        super().__init__()

        self.camera_n = len(camera)
        self.camera = camera

        self.st_gcn_networks = nn.ModuleList()
        for i in range(self.camera_n):
            self.st_gcn_networks.append(STGCN_Network(in_channels, num_class, graph_args,
                edge_importance_weighting, attention, **kwargs))
            
        # fusion for different camera
        self.fusion = nn.Sequential(nn.Linear(self.camera_n, self.camera_n), 
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(self.camera_n, 1))

        # prediction for total, hand, arm, sholder, depth, rate and release
        self.fcn = FCN(256, num_class, upper_bound)

    def forward(self, x):
        # just keep the body part
        x = x[:, :, :, :, 11:]
        # just keep the camera we need
        x = x[:, self.camera, :, :, :]
        # from [N, Camera, T, C, V] to [Camera, N, C, T, V, M]
        x = x.permute(1, 0, 3, 2, 4).contiguous()
        x = x.unsqueeze(-1)

        # get the output of each camera
        output = []
        for i, gcn in enumerate(self.st_gcn_networks):
            tmp_output = gcn(x[i])
            output.append(tmp_output)
         
        output = torch.stack(output, dim = 0)
        # output: [Camera, N, C] -> [N, C, camera_n])
        output = output.permute(1, 2, 0)
        # fusion
        output = self.fusion(output)
        output = output.squeeze(-1)
        # prediction for total, hand, arm, sholder, depth, rate and release
        total, hand_pos, arm_pos, sholder_pos, depth, rate, release = self.fcn(output)

        return total, hand_pos, arm_pos, sholder_pos, depth, rate, release


class FCN(nn.Module):
    
    def __init__(self, in_channels, num_class, upper_bound = 4):

        super().__init__()

        # fcn for prediction for total, hand, arm, sholder, depth, rate and release
        self.fcn_hand = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, num_class))
        self.fcn_arm = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, num_class))
        self.fcn_sholder = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, num_class))
        self.fcn_depth = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, num_class))
        self.fcn_rate = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, num_class))
        self.fcn_release = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, num_class))

        self.upper_bound = upper_bound

    def forward(self, x):

        # prediction
        hand_pos = self.fcn_hand(x)
        arm_pos = self.fcn_arm(x)
        sholder_pos = self.fcn_sholder(x)
        depth = self.fcn_depth(x)
        rate = self.fcn_rate(x)
        release = self.fcn_release(x)
        # set upper bound
        hand_pos = torch.clamp(hand_pos, 0, self.upper_bound)
        arm_pos = torch.clamp(arm_pos, 0, self.upper_bound)
        sholder_pos = torch.clamp(sholder_pos, 0, self.upper_bound)
        depth = torch.clamp(depth, 0, self.upper_bound)
        rate = torch.clamp(rate, 0, self.upper_bound)
        release = torch.clamp(release, 0, self.upper_bound)
        # total
        total = hand_pos + arm_pos + sholder_pos + depth + rate + release
        return total, hand_pos, arm_pos, sholder_pos, depth, rate, release   


class STGCN_Network(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, attention, **kwargs):
        super().__init__()



        # load graph
        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=True, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # initialize parameters for attention
        self.attention = attention
        self.attention_m = attention_m()


        

    def forward(self, x):

        if self.attention:
            # attention
            attention_descriptor = torch.mean(x,dim=(1,2),keepdim=True)
            #print(attention_descriptor.size())
            attention_descriptor = self.attention_m(attention_descriptor)
            x = attention_descriptor*x
            self.attention_descriptor = attention_descriptor

        # data normalization
        N, C, T, V, M = x.size()
        # x = x.permute(0, 4, 3, 1, 2).contiguous()
        # x = x.view(N * M, V * C, T)
        # x = self.data_bn(x)
        # x = x.view(N, M, V, C, T)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = x.squeeze(-1).squeeze(-1)
        
        return x

class attention_m(nn.Module):

    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(25, 25),
            nn.ReLU(inplace=True),
            nn.Linear(25, 25),
            nn.Sigmoid(),)

    def forward(self, x):
        
        N, C, T, V, M = x.size()
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        x = self.m(x)
        x = x.permute(0, 1, 2, 4, 3).contiguous()

        return x

class st_gcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A