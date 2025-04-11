import torch
import torch.nn as nn
import tinycudann as tcnn
from model.activation import trunc_exp
from model.renderer import LiDAR_Renderer
from model.planes_field import Planes4D
from model.hash_field import HashGrid4D
from model.flow_field import FlowField
from model.unet import UNet
from model.ResNet import ResNet_34


class SNLiDAR(LiDAR_Renderer):
    def __init__(
        self,
        min_resolution=32,
        base_resolution=512,
        max_resolution=32768,
        time_resolution=8,
        n_levels_plane=4,
        n_features_per_level_plane=8,
        n_levels_hash=8,
        n_features_per_level_hash=4,
        log2_hashmap_size=19,
        sem_feat_dim=128,
        num_layers_flow=3,
        hidden_dim_flow=64,
        num_layers_sigma=2,
        hidden_dim_sigma=64,
        geo_feat_dim=15,
        num_layers_lidar=3,
        hidden_dim_lidar=64,
        out_lidar_dim=2,
        num_frames=51,
        bound=1,
        num_semantic_classes=20,
        hidden_dim_semantic=64,
        **kwargs,
    ):
        super().__init__(bound, **kwargs)

        self.out_lidar_dim = out_lidar_dim
        self.num_frames = num_frames
        self.num_semantic_classes = num_semantic_classes
        
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)

        self.planes_encoder = Planes4D(
            grid_dimensions=2,
            input_dim=4,
            output_dim=n_features_per_level_plane,
            resolution=[min_resolution] * 3 + [time_resolution],
            multiscale_res=[2**(n) for n in range(n_levels_plane)],
        )

        self.hash_encoder = HashGrid4D(
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            time_resolution=time_resolution,
            n_levels=n_levels_hash,
            n_features_per_level=n_features_per_level_hash,
            log2_hashmap_size=log2_hashmap_size,
        )
        
        self.view_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "degree": 12,
            },
        )

        self.flow_net = FlowField(
            input_dim=4,
            num_layers=num_layers_flow,
            hidden_dim=hidden_dim_flow,
            use_grid=True,
        )

        self.intensity_net = tcnn.Network(
            n_input_dims=self.view_encoder.n_output_dims + geo_feat_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_lidar,
                "n_hidden_layers": num_layers_lidar - 1,
            },
        )

        self.raydrop_net = tcnn.Network(
            n_input_dims=self.view_encoder.n_output_dims + geo_feat_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_lidar,
                "n_hidden_layers": num_layers_lidar - 1,
            },
        )
        
        # load pretrained rangenet
        self.semantic_encoder = ResNet_34(nclasses=20, aux=True).cuda()
        def convert_relu_to_softplus(model, act):
            for child_name, child in model.named_children():
                if isinstance(child, nn.LeakyReLU):
                    setattr(model, child_name, act)
                else:
                    convert_relu_to_softplus(child, act)
        convert_relu_to_softplus(self.semantic_encoder, nn.Hardswish())
        pretrained_weights = torch.load('model/SalsaNext')
        pretrained_weights = pretrained_weights['state_dict']
        pretrained_conv1_weight = pretrained_weights['conv1.conv.weight']
        # modify input channel to 1
        new_conv1_weight = pretrained_conv1_weight.mean(dim=1, keepdim=True)
        pretrained_weights['conv1.conv.weight'] = new_conv1_weight
        self.semantic_encoder.load_state_dict(pretrained_weights, strict=False)
        # replace output layers
        nclasses = self.num_semantic_classes
        self.semantic_encoder.semantic_output = nn.Conv2d(128, nclasses, 1)
        self.semantic_encoder.aux_head1 = nn.Conv2d(128, nclasses, 1)
        self.semantic_encoder.aux_head2 = nn.Conv2d(128, nclasses, 1)
        self.semantic_encoder.aux_head3 = nn.Conv2d(128, nclasses, 1)
        
        self.latent_size = sem_feat_dim
        
        plane_hash_dim = self.planes_encoder.n_output_dims + self.hash_encoder.n_output_dims
        self.semantic_net = tcnn.Network(
            n_input_dims=plane_hash_dim+ self.latent_size,
            n_output_dims=self.num_semantic_classes,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_semantic,
                "n_hidden_layers": num_layers_lidar - 1,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=self.planes_encoder.n_output_dims + self.hash_encoder.n_output_dims + self.latent_size,
            n_output_dims=1 + geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_sigma,
                "n_hidden_layers": num_layers_sigma - 1,
            },
        )
        
        unet_in_channels = 3
        self.unet = UNet(in_channels=unet_in_channels, out_channels=1)

    def forward(self, x, d, t):
        pass

    def flow(self, x, t):
        # x: [N, 3] in [-bound, bound] for point clouds
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]

        if t.shape[0] == 1:
            t = t.repeat(x.shape[0], 1)
        xt = torch.cat([x, t], dim=-1) # xt: [N, 4]

        flow = self.flow_net(xt) # flow: [N, 6]

        return {
            "forward": flow[:, :3],
            "backward": flow[:, 3:],
        }

    def density(self, x, t=None, use_sem_feat=False, rays_index=None):
        # x: [N, 3], in [-bound, bound]
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        frame_idx = int(t * (self.num_frames - 1))

        hash_feat_s, hash_feat_d = self.hash_encoder(x, t)

        if t.shape[0] == 1:
            t = t.repeat(x.shape[0], 1)
        xt = torch.cat([x, t], dim=-1)

        plane_feat_s, plane_feat_d = self.planes_encoder(xt)

        # integrate neighboring dynamic features
        flow = self.flow_net(xt)
        hash_feat_1 = hash_feat_2 = hash_feat_d
        plane_feat_1 = plane_feat_2 = plane_feat_d
        if frame_idx < self.num_frames - 1:
            # forward t1 > t
            x1 = x + flow[:, :3]
            t1 = torch.tensor((frame_idx + 1) / self.num_frames)
            with torch.no_grad():
                hash_feat_1 = self.hash_encoder.forward_dynamic(x1, t1)
            t1 = t1.repeat(x1.shape[0], 1).to(x1.device)
            xt1 = torch.cat([x1, t1], dim=-1)
            plane_feat_1 = self.planes_encoder.forward_dynamic(xt1)

        if frame_idx > 0:
            # backward t2 < t
            x2 = x + flow[:, 3:]
            t2 = torch.tensor((frame_idx - 1) / self.num_frames)
            with torch.no_grad():
                hash_feat_2 = self.hash_encoder.forward_dynamic(x2, t2)
            t2 = t2.repeat(x2.shape[0], 1).to(x2.device)
            xt2 = torch.cat([x2, t2], dim=-1)
            plane_feat_2 = self.planes_encoder.forward_dynamic(xt2)

        plane_feat_d = 0.5 * plane_feat_d + 0.25 * (plane_feat_1 + plane_feat_2)
        hash_feat_d = 0.5 * hash_feat_d + 0.25 * (hash_feat_1 + hash_feat_2)

        plane_hash_features = torch.cat([plane_feat_s, plane_feat_d,
                              hash_feat_s, hash_feat_d], dim=-1)
        if use_sem_feat:
            semantic_feat = self.encode(x, rays_index)
        else:
            semantic_feat = torch.zeros(x.shape[0], self.latent_size, dtype=x.dtype, device=x.device)
        mlp_input = torch.cat([plane_hash_features, semantic_feat], dim=-1)
        sigma_semantic = self.semantic_net(mlp_input)
        
        h = self.sigma_net(mlp_input)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            "sigma": sigma,
            "geo_feat": geo_feat,
            "plane_hash_feat": plane_hash_features,
            "sigma_semantic": sigma_semantic,
        }
    
    def encoder_init(self, images):
        """
        :param images (NS, 1, H, W)
        NS is number of input (aka source or reference) views
        """
        self.num_views_per_obj = images.shape[0] if len(images.shape) == 4 else 1
        self.image_shape = torch.tensor(images.shape[-2:])
        _, res_4 = self.semantic_encoder(images)
        self.current_sem_feat = res_4 # (NS, 128, H, W)
        
    def encode(self, xyz, rays_index, coarse=True, viewdirs=None, far=False):
        """
        Please call encoder_init first!
        :param xyz (B, 3)
        B is batch of points (in rays)
        :param rays_index (NV, num_rays)
        :return (NS * B, latent) latent
        NS is number of input views
        """
        B, _ = xyz.shape
        NS = self.num_views_per_obj

        # Grab encoder's latent code.
        u = rays_index // self.image_shape[0]
        u = u.view(-1).int()
        v = rays_index % self.image_shape[0]
        v = v.view(-1).int()
        semantic_feat = self.current_sem_feat[:, :, v, u]
        semantic_feat = semantic_feat.squeeze(0).permute(1, 0).repeat(B // rays_index.shape[-1], 1)

        return semantic_feat
    
    # allow masked inference
    def attribute(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        
        if mask is not None:
            output = torch.zeros(
                mask.shape[0], self.out_lidar_dim-self.num_semantic_classes, dtype=x.dtype, device=x.device
            )
            # in case of empty mask
            if not mask.any():
                return output
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = (d + 1) / 2  # to [0, 1]
        d = self.view_encoder(d)

        intensity = self.intensity_net(torch.cat([d, geo_feat], dim=-1))
        intensity = torch.sigmoid(intensity)

        raydrop = self.raydrop_net(torch.cat([d, geo_feat], dim=-1))
        raydrop = torch.sigmoid(raydrop)
        
        h = torch.cat([raydrop, intensity], dim=-1) # [N, 2]

        if mask is not None:
            output[mask] = h.to(output.dtype)  # fp16 --> fp32
        else:
            output = h

        return output

    # optimizer utils
    def get_params(self, lr):
        params = [
            {"params": self.planes_encoder.parameters(), "lr": lr},
            {"params": self.hash_encoder.parameters(), "lr": lr},
            {"params": self.view_encoder.parameters(), "lr": lr},
            {"params": self.flow_net.parameters(), "lr": 0.1 * lr},       
            {"params": self.sigma_net.parameters(), "lr": 0.1 * lr},
            {"params": self.intensity_net.parameters(), "lr": 0.1 * lr},
            {"params": self.raydrop_net.parameters(), "lr": 0.1 * lr},
        ]

        return params