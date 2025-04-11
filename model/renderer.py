import torch
import torch.nn as nn


class LiDAR_Renderer(nn.Module):
    def __init__(
        self,
        bound=1,
        near_lidar=0.01,
        far_lidar=0.81,
        density_scale=1,
        active_sensor=False,
    ):
        super().__init__()
        self.bound = bound
        self.near_lidar = near_lidar
        self.far_lidar = far_lidar
        self.density_scale = density_scale
        self.active_sensor = active_sensor

        aabb = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound]) # axis-aligned bounding box
        self.register_buffer("aabb", aabb)

    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and intensity/raydrop query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def encoder_init(self, images, poses, focal, z_bounds, c):
        raise NotImplementedError()
    
    def encode(self, xyz, coarse, viewdirs, far):
        raise NotImplementedError()

    def attribute(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def run(
        self,
        rays_o,
        rays_d,
        rays_index,
        time, 
        num_steps=768,
        perturb=False,
        use_sem_feat=False,
        **kwargs
    ):
        out_lidar_dim = self.out_lidar_dim-self.num_semantic_classes

        prefix = rays_o.shape[:-1] # B, N
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]
        device = rays_o.device

        aabb = self.aabb

        # hard code
        nears = torch.ones(N, dtype=rays_o.dtype, device=rays_o.device) * self.near_lidar
        fars = torch.ones(N, dtype=rays_o.dtype, device=rays_o.device) * self.far_lidar

        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)
        z_vals = z_vals.expand((N, num_steps))
        z_vals = nears + (fars - nears) * z_vals

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])

        # query SDF and RGB
        density_outputs = self.density(
            xyzs.reshape(-1, 3),
            time,
            use_sem_feat,
            rays_index
        )

        for k, v in density_outputs.items():
            if density_outputs[k] is not None:
                density_outputs[k] = v.view(N, num_steps, -1)

        deltas = z_vals[..., 1:] - z_vals[..., :-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs["sigma"].squeeze(-1))
        if self.active_sensor:
            alphas = 1 - torch.exp(-2 * deltas * self.density_scale * density_outputs["sigma"].squeeze(-1))
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
        
        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            if density_outputs[k] is not None:
                density_outputs[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4  # hard coded
        
        attr =  self.attribute(
            xyzs.reshape(-1, 3),
            dirs.reshape(-1, 3),
            mask=mask.reshape(-1),
            **density_outputs
        )
        attr = attr.view(N, -1, out_lidar_dim)

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)

        # calculate depth  Note: not real depth!!
        # ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        # depth = torch.sum(weights * ori_z_vals, dim=-1)
        depth = torch.sum(weights * z_vals, dim=-1)
        semantic = torch.sum(weights.unsqueeze(-1) * density_outputs["sigma_semantic"].view(N,-1,self.num_semantic_classes), dim=-2)

        # calculate lidar attributes
        image = torch.sum(weights.unsqueeze(-1) * attr, dim=-2)
        if self.use_semantic:
            image = torch.cat([image, semantic], dim=-1)

        image = image.view(*prefix, self.out_lidar_dim)
        depth = depth.view(*prefix)

        return {
            "depth_lidar": depth,
            "image_lidar": image,
            "weights_sum_lidar": weights_sum,
            "weights": weights,
            "z_vals": z_vals,
        }

    def render(
        self,
        rays_o,
        rays_d,
        rays_index,
        time,
        staged=False,
        max_ray_batch=4096,
        use_sem_feat=False,
        **kwargs
    ):
        _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        if staged:
            out_lidar_dim = self.out_lidar_dim
            res_keys = ["depth_lidar", "image_lidar"]
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, out_lidar_dim), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(
                        rays_o[b : b + 1, head:tail],
                        rays_d[b : b + 1, head:tail],
                        rays_index[b : b + 1, head:tail],
                        time[b:b+1],
                        use_sem_feat=use_sem_feat,
                        **kwargs
                    )
                    depth[b : b + 1, head:tail] = results_[res_keys[0]]
                    image[b : b + 1, head:tail] = results_[res_keys[1]]
                    head += max_ray_batch

            results = {}
            results[res_keys[0]] = depth
            results[res_keys[1]] = image

        else:
            results = _run(rays_o, rays_d, rays_index, time, use_sem_feat=use_sem_feat, **kwargs)

        return results
