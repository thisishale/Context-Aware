import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor_speed(nn.Module):

    def __init__(self, args):
        super(FeatureExtractor_speed, self).__init__()

        self.embbed_size = args.hidden_size_sp
        self.box_embed = nn.Sequential(nn.Linear(1, self.embbed_size),
                                        nn.LeakyReLU())
                                        # nn.ReLU())
    def forward(self, inputs):
        embedded_box_input= self.box_embed(inputs)
        return embedded_box_input

class FeatureExtractor_traj(nn.Module):

    def __init__(self, args):
        super(FeatureExtractor_traj, self).__init__()

        self.embbed_size = args.hidden_size_traj
        self.box_embed = nn.Sequential(nn.Linear(4, self.embbed_size),
                                        nn.LeakyReLU())
                                     # nn.ReLU()) 
    def forward(self, inputs):
        embedded_box_input= self.box_embed(inputs)
        return embedded_box_input


_FEATURE_EXTRACTORS = {
    'speed': FeatureExtractor_speed,
    'traj': FeatureExtractor_traj
}

def build_feature_extractor(args, ft_type):
    func = _FEATURE_EXTRACTORS[ft_type]
    return func(args)
