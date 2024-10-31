import torch 
import numpy as np 

# class SpecMixup:
#     def __init__(self,
#                  alpha=1.0, 
#                  prob=1.0,
#                  num_mix=2
#                  ):
#         self.alpha = alpha
#         self.prob = prob
#         self.num_mix = num_mix

#     def __call__(self, x, target):
#         x, target = self._mix_batch(x, target)
#         return x, target
    
#     def _mix_batch(self, x, target):

#         batch_size = x.size(0)
#         # which samples to apply mixup to 
#         apply_mixup = np.random.rand(batch_size) < self.prob


#         # Initialize mixed inputs and targets
#         x_mix = x.clone()
#         target = torch.tensor(target, dtype=torch.float32)
#         target_mix = target.clone()

#         for i in range(batch_size):
#             if apply_mixup[i]:
#                 # Generate mixing coefficients from a Dirichlet distribution
#                 mix_weights = np.random.dirichlet([self.alpha] * self.num_mix)
#                 mix_weights = torch.from_numpy(mix_weights).float()  # Shape: (num_mix,)

#                 # Randomly select indices for mixing, including the current sample
#                 mix_indices = torch.randperm(batch_size)[:self.num_mix]
#                 # Ensure the current sample is included
#                 if i not in mix_indices:
#                     mix_indices[0] = i

#                 # Mix inputs
#                 x_mix_i = mix_weights[0] * x[mix_indices[0]]
#                 for j in range(1, self.num_mix):
#                     x_mix_i += mix_weights[j] * x[mix_indices[j]]
#                 x_mix[i] = x_mix_i

#                 # Mix targets
#                 target_mix_i = mix_weights[0] * target[mix_indices[0]]
#                 for j in range(1, self.num_mix):
#                     target_mix_i += mix_weights[j] * target[mix_indices[j]]
#                 target_mix[i] = target_mix_i
#             else:
#                 # If Mixup is not applied, keep the original target
#                 target_mix[i] = target[i]
#         return x_mix, target_mix.tolist()


class SpecMixup:
    def __init__(self, alpha=1.0, prob=1.0, num_mix=2, full_target=False):
        self.alpha = alpha
        self.prob = prob
        self.num_mix = num_mix
        self.full_target = full_target

    def __call__(self, x, target):
        return self._mix_batch(x, target)
    
    def _mix_batch(self, x, target):
        batch_size = x.size(0)
        device = x.device
        is_waveform = len(x.shape) == 2  # True for waveforms, False for spectrograms

        # Determine which samples to apply mixup to
        apply_mixup = torch.rand(batch_size, device=device) < self.prob

        if not apply_mixup.any():
            return x, target

        # Convert target to tensor if it's not already
        target = torch.tensor(target, dtype=torch.float32, device=device)

        # Generate mixing coefficients from a Dirichlet distribution
        mix_weights = torch.from_numpy(
            np.random.dirichlet([self.alpha] * self.num_mix, size=batch_size)
        ).float().to(device)

        # Generate random indices for mixing, excluding self-indices
        mix_indices = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, self.num_mix)
        for i in range(batch_size):
            pool = torch.cat([torch.arange(0, i), torch.arange(i+1, batch_size)])
            mix_indices[i, 1:] = pool[torch.randperm(batch_size-1)[:self.num_mix-1]]
            #mix_indices[i, 1:] = 14

        # Perform mixup
        x_mix = torch.zeros_like(x)
        target_mix = torch.zeros_like(target)
        
        for i in range(self.num_mix):
            if is_waveform:
                x_mix += apply_mixup.unsqueeze(1) * mix_weights[:, i].unsqueeze(1) * x[mix_indices[:, i]]
            else:
                x_mix += apply_mixup.unsqueeze(1).unsqueeze(2) * mix_weights[:, i].unsqueeze(1).unsqueeze(2) * x[mix_indices[:, i]]
            
            if self.full_target:
                # For full_target, use hard labels
                target_mix = torch.max(target_mix, target[mix_indices[:, i]])
            else:
                # For soft labels, use weighted sum
                target_mix += apply_mixup.unsqueeze(1) * mix_weights[:, i].unsqueeze(1) * target[mix_indices[:, i]]

        # Only replace mixed samples
        if is_waveform:
            x = torch.where(apply_mixup.unsqueeze(1), x_mix, x)
        else:
            x = torch.where(apply_mixup.unsqueeze(1).unsqueeze(2), x_mix, x)
        target = torch.where(apply_mixup.unsqueeze(1), target_mix, target)

        return x, target.tolist()