from pathlib import Path
from datetime import datetime
from functools import partial
import numpy as np
import math
import random
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from util.pos_embed import get_2d_sincos_pos_embed_flexible
import lightning as L
import hydra
from timm.optim.optim_factory import param_groups_weight_decay
from transformers import get_cosine_schedule_with_warmup


class EAT(L.LightningModule):
    def __init__(self, 
                 norm_layer,
                 mask_ratio,
                 cfg_encoder,
                 cfg_decoder,
                 cfg_teacher,
                 cfg_teacher_assistant,
                 cfg_task,
                 optimizer, 
                 scheduler,
                 compile_mode="default",  # None, "default", "reduce-overhead" -> takes more GPU
                ):
                    
        super().__init__()
        self.save_hyperparameters()

        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.mask_ratio = mask_ratio
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.train_batch_size = optimizer.train_batch_size
        self.layer_decay = optimizer.layer_decay
        self.clip_norm = optimizer.clip_norm
        self.use_ema_scheduler = cfg_teacher.use_ema_scheduler

        self.cfg_encoder = cfg_encoder
        self.cfg_decoder = cfg_decoder
        self.cfg_teacher = cfg_teacher
        self.cfg_teacher_assistant = cfg_teacher_assistant
        self.cfg_task = cfg_task

        self.cls_task_is_clustering = cfg_task.cls_task == 'clustering' 

        activation = nn.GELU

        if cfg_decoder.name == "CNN2d":
            decoder_cls = CNN2dDecoder
            decoder_kwargs = {
                'kernel_size': cfg_decoder.kernel_size,
                'stride': cfg_decoder.stride,
                'padding': cfg_decoder.padding,
                'groups': cfg_decoder.groups,
                'activation': activation,
                'num_layers': cfg_decoder.num_layers,
            }
        elif cfg_decoder.name == "MLP_LSTM":
            decoder_cls = MLP_LSTM_Decoder
            decoder_kwargs = {
                'drop': cfg_decoder.drop,
                'activation': activation,
                'bidirectional': cfg_decoder.bidirectional,
                'add_residual': cfg_decoder.add_residual,
                'num_layers': cfg_decoder.num_layers,
            }
        else:
            raise ValueError(f"Decoder name {cfg_decoder.name} not supported")

        if cfg_task.cls_task == 'clustering':
            assert cfg_task.clustering_regularizer in ['centering', 'sinkhornknopp', 'gini']            
            self.dino_loss = DINOLoss(cfg_task.num_clusters)  # num_clusters = 65536
            dinohead_kwargs = {'out_dim': cfg_task.num_clusters, 'use_bn': False, 'nlayers': 3, 'hidden_dim': 2048, 'bottleneck_dim': 256, 'mlp_bias': True}
        elif cfg_task.cls_task == 'regression':
            self.dino_loss = dinohead_kwargs = None
        else:
            raise ValueError(f"CLS task {cfg_task.cls_task} is not supported")
    
        if cfg_task.feature_regularizer is not None:  # assert feature_regularizer in ['koleo', 'var']
            if cfg_task.feature_regularizer == 'var':
                self.feature_regularizer_fn = var_loss
            elif cfg_task.feature_regularizer == 'koleo':
                self.feature_regularizer_fn = KoLeoLoss()
            else:
                raise ValueError(f"Feature regularizer {cfg_task.feature_regularizer} is not supported")
    
        # build student model
        self.student = EAT_Student(input_shape=(cfg_encoder.input_shape_t, cfg_encoder.input_shape_f),
                                   patch_size=(cfg_encoder.patch_size, cfg_encoder.patch_size),
                                   embed_dim=cfg_encoder.embed_dim,
                                   depth=cfg_encoder.depth,
                                   num_heads=cfg_encoder.num_heads,
                                   mlp_ratio=cfg_encoder.mlp_ratio,
                                   qkv_bias=cfg_encoder.qkv_bias,
                                   drop=cfg_encoder.drop,
                                   drop_path_rate=cfg_encoder.drop_path_rate,
                                   pos_trainable=cfg_encoder.pos_trainable,
                                   clone_size=cfg_encoder.clone_size, 
                                   mask_mode=cfg_encoder.mask_mode, 
                                   cls_task=cfg_task.cls_task,
                                   dinohead_kwargs=dinohead_kwargs,
                                   decoder_cls=decoder_cls,
                                   decoder_kwargs=decoder_kwargs,
                                  )
        
        # build teacher model
        self.teacher = EAT_Teacher(input_shape=(cfg_encoder.input_shape_t, cfg_encoder.input_shape_f),
                                   patch_size=(cfg_encoder.patch_size, cfg_encoder.patch_size),
                                   embed_dim=cfg_encoder.embed_dim,
                                   depth=cfg_encoder.depth,
                                   num_heads=cfg_encoder.num_heads,
                                   mlp_ratio=cfg_encoder.mlp_ratio,
                                   qkv_bias=cfg_encoder.qkv_bias,
                                   drop=cfg_encoder.drop,
                                   drop_path_rate=cfg_encoder.drop_path_rate,
                                   pos_trainable=cfg_encoder.pos_trainable,
                                   clone_size=cfg_encoder.clone_size,
                                   cls_task=cfg_task.cls_task,
                                   dinohead_kwargs=dinohead_kwargs,
                                   average_top_k_layers=cfg_teacher.average_top_k_layers,
                                   instance_norm_target_layer=cfg_teacher.instance_norm_target_layer,
                                   batch_norm_target_layer=cfg_teacher.batch_norm_target_layer,
                                   layer_norm_target_layer=cfg_teacher.layer_norm_target_layer,
                                   layer_norm_targets=cfg_teacher.layer_norm_targets,
                                   instance_norm_targets=cfg_teacher.instance_norm_targets,
                                  )
        
        self.teacher.encoder.load_state_dict(self.student.encoder.state_dict())
        if self.cls_task_is_clustering:
            self.teacher.head.load_state_dict(self.student.head.state_dict())
        self.teacher.requires_grad_(False)
        if self.cls_task_is_clustering and cfg_task.clustering_regularizer == 'gini':
            self.teacher.head.requires_grad_(True)

        if compile_mode is not None:
            print(f"Compiling student and teacher with mode {compile_mode}")
            self.student.compile(mode=compile_mode)
            self.teacher.compile(mode=compile_mode)
        
        if cfg_task.use_teacher_assistant:
            self.teacher_assistant = EAT_Teacher(input_shape=(cfg_teacher_assistant.input_shape_t, cfg_teacher_assistant.input_shape_f),
                                                 patch_size=(cfg_teacher_assistant.patch_size, cfg_teacher_assistant.patch_size),
                                                 embed_dim=cfg_teacher_assistant.embed_dim,
                                                 depth=cfg_teacher_assistant.depth,
                                                 num_heads=cfg_teacher_assistant.num_heads,
                                                 mlp_ratio=cfg_teacher_assistant.mlp_ratio,
                                                 qkv_bias=cfg_teacher_assistant.qkv_bias,
                                                 drop=cfg_teacher_assistant.drop,
                                                 drop_path_rate=cfg_teacher_assistant.drop_path_rate,
                                                 pos_trainable=False,
                                                 clone_size=cfg_encoder.clone_size,
                                                 cls_task='regression',
                                                 dinohead_kwargs=None,
                                                 average_top_k_layers=cfg_teacher.average_top_k_layers,
                                                 instance_norm_target_layer=cfg_teacher.instance_norm_target_layer,
                                                 batch_norm_target_layer=cfg_teacher.batch_norm_target_layer,
                                                 layer_norm_target_layer=cfg_teacher.layer_norm_target_layer,
                                                 layer_norm_targets=cfg_teacher.layer_norm_targets,
                                                 instance_norm_targets=cfg_teacher.instance_norm_targets,
                                                )
            self.teacher_assistant.encoder = load_eat_audioset_pretrained_state(self.teacher_assistant.encoder, audioset_eat_state_path=cfg_teacher_assistant.audioset_state_path)
            self.teacher_assistant.requires_grad_(False)
            self.teacher_assistant.compile(mode=compile_mode)
        else:
            self.teacher_assistant = None
    
        self.ema_scheduler = None
        self.sigmoid_scheduler = None
        self.training_step_count = 0 
    
    def forward(self, x, mask_ratio=None):
        """
        args:
            x - mel-spectrogram of shape (batch, channels=1, time=512, freq=128)
            mask_ratio - masking percentage in range (0, 1)
        """
        
        self.student.train()
        self.teacher.eval()
        if self.cfg_task.clustering_regularizer == "gini":
            self.teacher.head.train()
        if self.teacher_assistant is not None:
            self.teacher_assistant.eval()

        if mask_ratio is None:
            mask_ratio = self.mask_ratio 

        cache = {}
        
        # student output shapes: (B=batch_size*clone_size, D=768), (B, L=256, D), (B, L)
        cache['student_cls_tokens'], cache['student_patch_tokens'], cache['mask'] = self.student(x, mask_ratio)

        # ALWAYS get teacher outputs (needed for both clustering and regression)
        with torch.no_grad():
            teacher_cls_tokens, cache['teacher_patch_tokens'] = self.teacher(x)
            if self.teacher_assistant is not None:
                _, cache['assistant_patch_tokens'] = self.teacher_assistant(x)
        
        # Only do clustering-specific processing if clustering task
        if self.cls_task_is_clustering:
            cache['student_cls_tokens_after_head'] = self.student.head(cache['student_cls_tokens'])
            cache['teacher_cls_tokens_after_head'] = self.teacher.head(teacher_cls_tokens)
        
        return cache
    
    def training_step(self, batch, batch_idx):
        audio = batch["audio"]

        # forward
        cache = self(audio, self.mask_ratio)
        
        mask = cache['mask'].float()  # masked patches are 1 and visible ones are 0 (in student input)
        student_cls_tokens = cache['student_cls_tokens'].float()  # cls token from last student layer
        student_patch_tokens = cache['student_patch_tokens'].float()  # other token from last student layer
        teacher_patch_tokens = cache['teacher_patch_tokens'].float()  # average of tokens across layers without cls

        if self.cls_task_is_clustering:  # this is for clustering (DINO)
            teacher_cls_tokens_after_head = cache['teacher_cls_tokens_after_head'].float()  # B, C
            student_cls_tokens_after_head = cache['student_cls_tokens_after_head'].float()  # B, C
        else:  # the usual EAT cls loss, overwrite it with the mean of patch tokens
            teacher_cls_tokens = teacher_patch_tokens.mean(dim=1)  # B, D

        # we only use the teacher assistant for regression tasks of EAT
        if self.teacher_assistant is not None:
            assistant_patch_tokens = cache['assistant_patch_tokens'].float()  # B, L, D
            alpha = self.sigmoid_scheduler.step()  # alpha gradually goes from ~0 -> 1
            teacher_patch_tokens = alpha * teacher_patch_tokens + (1. - alpha) * assistant_patch_tokens
            if not self.cls_task_is_clustering:  # usual EAT loss for the cls
                assistant_cls_tokens = assistant_patch_tokens.mean(dim=1)  # B, D
                teacher_cls_tokens = alpha * teacher_cls_tokens + (1. - alpha) * assistant_cls_tokens

        total_loss, loss_dict = 0, {}
        
        # EAT local loss (maked patch reconstruction in latent space); it is similar to iBOT
        patch_loss = masked_reconstruction_loss(student_patch_tokens, teacher_patch_tokens, mask)
        total_loss += patch_loss
        loss_dict['train_patch_loss'] = patch_loss  #.clone().detach().cpu().item()
        
        if self.cls_task_is_clustering: # Dino clustering Loss
            
            if self.cfg_task.clustering_regularizer == "centering":
                teacher_cls_tokens_probs = self.dino_loss.softmax_center_teacher(teacher_cls_tokens_after_head, teacher_temp=0.07)
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
            
            elif self.cfg_task.clustering_regularizer == "sinkhornknopp":
                teacher_cls_tokens_probs = self.dino_loss.sinkhorn_knopp_teacher(teacher_cls_tokens_after_head, teacher_temp=0.07)
            
            elif self.cfg_task.clustering_regularizer == "gini":
                teacher_cls_tokens_probs = F.softmax(teacher_cls_tokens_after_head, dim=1)
                # maximize impurity for uniform cluster assignment to prevent collaps
                gini = self.dino_loss.gini_impurity(teacher_cls_tokens_probs)
                total_loss -= gini
                loss_dict['gini'] = gini  #.clone().detach().cpu().item()
            
            else:
                raise ValueError('clustering regularizer should be one of [centering, sinkhornknopp, gini].')
            
            student_temp = 1 if self.cfg_task.clustering_regularizer == 'gini' else None
            cls_loss = self.dino_loss.forward(student_cls_tokens_after_head, teacher_cls_tokens_probs, student_temp)
        
        else:  # EAT regression loss
            cls_loss = torch.mean((student_cls_tokens - teacher_cls_tokens) ** 2.)

        total_loss += cls_loss
        loss_dict['train_cls_loss'] = cls_loss  #.clone().detach().cpu().item()
        
        # latent space diversity regularizer
        if self.feature_regularizer_fn is not None:
            cls_diversity_loss = self.feature_regularizer_fn(student_cls_tokens)
            total_loss += cls_diversity_loss
            loss_dict['train_cls_diversity_loss'] = cls_diversity_loss.clone().detach().cpu().item()
            if self.cfg_task.regularize_patch_tokens:
                patch_diversity_loss = self.feature_regularizer_fn(student_patch_tokens.flatten(0, 1))  # (B, L, D).flatten(0, 1) -> (B*L, D)
                total_loss += patch_diversity_loss
                loss_dict['train_patch_diversity_loss'] = patch_diversity_loss  #.clone().detach().cpu().item()

        loss_dict['train_loss'] = total_loss  #.clone().detach().cpu().item()
        
        # batch_size = audio.shape[0] * self.student.clone_size  
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True, batch_size=audio.size(0))
        self.training_step_count += 1
        
        return total_loss
            
    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.clip_norm)
        if self.cls_task_is_clustering and self.cfg_task.clustering_regularizer == 'gini':
            torch.nn.utils.clip_grad_norm_(self.teacher.head.parameters(), self.clip_norm)

        # EMA update teacher
        if self.ema_scheduler is not None:
            decay = self.ema_scheduler.step()
        else: 
            decay = 0.999
        ema_update(self.teacher.encoder, self.student.encoder, decay=decay)
        if self.cls_task_is_clustering and self.cfg_task.clustering_regularizer != 'gini':
            ema_update(self.teacher.head, self.student.head, decay=decay)

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        if self.optimizer_cfg.get("lr scaler", False):
            accum = int(self.trainer.accumulate_grad_batches)  
            world_size = self.trainer.num_devices * self.trainer.num_nodes
            clone_fac = getattr(self.student, "clone_size", 1)
            eff_batch = accum * world_size * self.train_batch_size * clone_fac

            base_lr = self.optimizer_cfg["lr"]            # 0.0005 from paper
            scaled_lr = base_lr * eff_batch / 768 # 768 = 12*16*4
            self.optimizer_cfg["lr"] = scaled_lr    
            print(f"Scaled LR: {scaled_lr}")    

        param_groups = param_groups_weight_decay(self.student, self.optimizer_cfg["weight_decay"], no_weight_decay_list=("bias", "bn", "ln", "gn", "norm"))

        if self.cfg_task.cls_task == 'clustering' and self.cfg_task.clustering_regularizer == 'gini':
            param_groups += param_groups_weight_decay(self.teacher.head, self.optimizer_cfg["weight_decay"], no_weight_decay_list=("bias", "bn", "ln", "gn", "norm"))
            
        optimizer = torch.optim.AdamW(param_groups, lr=self.optimizer_cfg["lr"], betas=self.optimizer_cfg["betas"])

        num_training_steps = self.trainer.estimated_stepping_batches

        # EMA scheduler
        if self.use_ema_scheduler: 
            steps_per_epoch = num_training_steps // self.trainer.max_epochs
            epochs_for_ema = self.trainer.max_epochs // 4 
            ema_max_iter = int(steps_per_epoch * epochs_for_ema)
            self.ema_scheduler = EMA_Weight_Decay_Scheduler(decay_start=0.9998, decay_end=0.99999, max_iter=ema_max_iter)
        
        if self.cfg_task.use_teacher_assistant:
            steps_per_epoch = num_training_steps // self.trainer.max_epochs
            epochs_for_ta = self.trainer.max_epochs // 1
            ta_max_iter = int(steps_per_epoch * epochs_for_ta)
            self.sigmoid_scheduler = Sigmoid_Rampup_Scheduler(max_iter=ta_max_iter) 
            
        if self.scheduler_cfg:
            if self.scheduler_cfg.get('use_custom_scheduler', False):
                raise NotImplementedError("Custom scheduler not implemented yet")
            else:
                warmup_ratio = 0.125
                num_warmup_steps = int(num_training_steps * warmup_ratio)
                scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
                scheduler_dict = {"scheduler": scheduler, "interval": "step", "frequency": 1}
                return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        else:
            return {"optimizer": optimizer}


# Decoder
class Conv2dLayerNorm(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding='same', groups=1, activation=nn.GELU, add_residual=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups)
        self.add_residual = add_residual
        self.ln = nn.LayerNorm(out_c, elementwise_affine=False)
        self.activation = activation()
        
    def forward(self, x):
        inp = x
        x = self.conv(x)
        x = x.transpose(-3, -1)
        x = self.ln(x)
        x = x.transpose(-3, -1)
        x = self.activation(x)
        if self.add_residual and x.size(1) == inp.size(1):
            x = x + inp
        return x

class CNN2dDecoder(nn.Module):
    def __init__(self, dim=768, kernel_size=3, stride=1, padding='same', groups=16, activation=nn.GELU, add_residual=True, num_layers=6, num_freq_patches=8, num_time_patches=32):
        super().__init__()
        self.blocks = nn.Sequential(*[Conv2dLayerNorm(dim, dim, kernel_size, stride, padding, groups, activation, add_residual) for _ in range(num_layers)])
        self.proj = nn.Linear(dim, dim, bias=True) # decoder to patch
        self.f = num_freq_patches
        self.t = num_time_patches
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
    def forward(self, x, ids_restore):
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore)
        x = x.transpose(-2, -1)  # B, D, L
        x = x.reshape(x.shape[0], x.shape[1], self.t, self.f)
        x = self.blocks(x)
        x = x.flatten(2).transpose(-2, -1)  # B, L, D
        return self.proj(x)

## MLP-LSTM
class MLP_LSTM_Block(nn.Module):
    
    def __init__(self, dim=768, drop=0., activation=nn.GELU, bidirectional=True, add_residual=True):
        super().__init__()
        self.add_residual = add_residual
        self.act = activation()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.ln1 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        self.lstm = nn.LSTM(dim, dim, batch_first=True, bidirectional=bidirectional)
        fc2_inp_size = int(dim * 2) if bidirectional else dim
        self.fc2 =  nn.Linear(fc2_inp_size, dim, bias=False)
        self.ln2 = nn.LayerNorm(dim)
                
    def forward(self, x):
        self.lstm.flatten_parameters()
        z = self.fc1(x)
        z = self.act(z)
        z = self.ln1(z)
        z = self.drop(z)
        with torch.autocast(device_type='cuda', enabled=False):  # device_type=x.device
            z = z.float()
            z, _ = self.lstm(z)
        z = self.fc2(z)
        z = self.act(z)
        out = self.ln2(z)
        if self.add_residual:
            out = out + x
        return out

class MLP_LSTM_Decoder(nn.Module):
    def __init__(self, dim=768, drop=0., activation=nn.GELU, bidirectional=True, add_residual=True, num_layers=2, **kwargs):
        super().__init__()
        self.blocks = nn.Sequential(*[MLP_LSTM_Block(dim, drop, activation, bidirectional, add_residual) for _ in range(num_layers)])
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
    def forward(self, x, ids_restore):
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore)
        return self.blocks(x)

# Vit Encoder for EAT
## https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py#L170
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, scale_norm=False, proj_bias=True, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        """
        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        # if qk_norm or scale_norm:
            # assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
      
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # attn = maybe_add_mask(attn, attn_mask)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AltBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_norm_first=False, ffn_targets=True):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rel_pos_bias=None, pos_mask=None):
        if self.layer_norm_first:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            t = self.mlp(self.norm2(x))
            x = x + self.drop_path(t)
            if not self.ffn_targets:
                t = x
            return x, t
        else:
            x = x + self.drop_path(self.attn(x))
            r = x = self.norm1(x)
            x = self.mlp(x)
            t = x
            x = self.norm2(r + self.drop_path(x))
            if not self.ffn_targets:
                t = x
            return x, t

class PatchEmbed(nn.Module):

    def __init__(self, img_size=(512, 128), patch_size=(16, 16), in_chans=1, emb_dim=768):
        super().__init__()
        assert isinstance(patch_size, tuple)
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 256
        self.patch_ft = (img_size[1] // patch_size[1], img_size[0] // patch_size[0]) # number of patches height/width = 8/32
        self.proj = nn.Conv2d(in_chans, emb_dim, kernel_size=patch_size, stride=patch_size)
       
    def forward(self, x):
        x = self.proj(x) # B, C=1, T=512, F=128 -> B, 768, 32, 8
        x = x.flatten(2) # B, 768, 32, 8 -> B, 768, 256
        x = x.transpose(1, 2) # B, 768, 256 -> B, 256, 768
        return x

class ViT_MaskedEncoder(nn.Module):
    def __init__(self, input_shape=(512, 128), patch_size=(16, 16), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop=0., attn_drop=0., drop_path_rate=0., pos_trainable=False, clone_size=16, mode='student', mask_mode='inv'):
        super().__init__()
        
        assert mode in ['student', 'teacher']
        assert mask_mode in ['rand', 'inv', 'seq']
        assert (input_shape[0] % patch_size[0]) == 0 and (input_shape[1] % patch_size[1]) == 0
        
        if mode == 'student':
            self.forward_fn = self.student_forward
            if mask_mode == 'rand':
                self.mask_fn = self.random_masking
            elif mask_mode == 'inv':
                self.mask_fn = self.inverse_block_mask
            else:
                self.mask_fn = self.sequential_mask
                assert patch_size[0] == 1 and patch_size[1] == input_shape[1]
        else:
            self.forward_fn = self.teacher_forward
            self.mask_fn = None

        self.clone_size = clone_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(input_shape, patch_size, 1, embed_dim)
        pos_embed = get_2d_sincos_pos_embed_flexible(embed_dim, self.patch_embed.patch_ft, cls_token=True)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=pos_trainable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02) # 1, 1, 768
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([AltBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def sequential_mask(self, shape, mask_ratio=0.8, num_sub_seq=4):
        
        if mask_ratio == 0:
            return None, None, None
                
        B, L, D = shape
        B = B * self.clone_size
        sub_seq_len = L // num_sub_seq
        sub_seq_mask_len = int(mask_ratio * sub_seq_len)
        sub_seq_mask_max_start_idx = sub_seq_len - sub_seq_mask_len

        # 1 are places to remove, 0 are places to keep.
        mask = torch.zeros((B, L))
        for b in range(B):
            for i in range(num_sub_seq):
                start = random.randint(0, sub_seq_mask_max_start_idx - 1) + i * sub_seq_len
                mask[b, start:start+sub_seq_mask_len] = 1
            
        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)
        
        len_keep = L - mask[0].sum()
        ids_keep = ids_shuffle[:, :len_keep]
        ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        
        return mask.float(), ids_keep, ids_restore
        
    def inverse_block_mask(self, shape, mask_ratio=0.8, mask_length=5, mask_prob_adjust=0.07, require_same_masks=True):
        
        if mask_ratio == 0:
            return None, None, None
        
        assert mask_length > 1
        
        num_freq_patches, num_time_patches = self.patch_embed.patch_ft
        B, L, D = shape
        B = B * self.clone_size
        d = (num_time_patches, num_freq_patches)
        mask_ratio = 1 - mask_ratio    
        
        mask = torch.zeros((B, d[0], d[1]))
        masking_size = int(L * ((mask_ratio + mask_prob_adjust) / (mask_length ** 2)))
        mask_inds = torch.randint(0, L, size=(B, masking_size))
        mask.view(B, -1).scatter_(1, mask_inds, 1)
        
        centers = mask.nonzero(as_tuple=True)
        inds = ([], [], [])
        offset = mask_length // 2
        for i in range(mask_length):
            for j in range(mask_length):
                k1 = i - offset
                k2 = j - offset
                inds[0].append(centers[0])
                inds[1].append(centers[1] + k1)
                inds[2].append(centers[2] + k2)
        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=d[0] - 1)
        i2 = torch.cat(inds[2]).clamp_(min=0, max=d[1] - 1)
        mask[(i0, i1, i2)] = 1
        mask = mask.reshape(B, -1)
        
        if require_same_masks:
            n_masks = mask.sum(dim=-1)
            target_len = int(L * (mask_ratio))
            for i in range(len(mask)):
                n = n_masks[i]
                m = mask[i]
                r = 0
                if n > target_len:
                    to_unmask = torch.multinomial(m, int(n - target_len), replacement=False)
                    m[to_unmask] = 0
                elif n < target_len:
                    to_mask = torch.multinomial((1 - m), int(target_len - n), replacement=False)
                    m[to_mask] = 1
                    
        # now inverse_mask: 1 are places to remove, 0 are places to keep.
        mask = 1 - mask  
        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)
        len_keep = L - mask[0].sum()
        ids_keep = ids_shuffle[:, :len_keep]
        ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        return mask.float(), ids_keep, ids_restore
    
    def random_masking(self, shape, mask_ratio=0.8):
        
        if mask_ratio == 0:
            return None, None, None
        
        B, L, D = shape  # batch, length, dim
        B *= self.clone_size
        
        len_keep = int(L * (1 - mask_ratio))    
        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = noise.argsort(dim=1)  # ascend: small is keep, large is remove
        ids_restore = ids_shuffle.argsort(dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        ids_restore = ids_restore.unsqueeze(-1).repeat(1, 1, D)
        ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        
        return mask, ids_keep, ids_restore
                
    def student_forward(self, x, mask_ratio=None):
        x = self.patch_embed(x)  # B, 1, T, F -> B, L=256, D=768
        x = x + self.pos_embed[:, 1:, :]
        
        # generate masks of shape: (B*clone_size, L)
        mask, ids_keep, ids_restore = self.mask_fn(x.shape, mask_ratio)
        mask, ids_keep, ids_restore = mask.to(x.device), ids_keep.to(x.device), ids_restore.to(x.device)

        # repeat the inputs for clone_size on batch axis
        x = x.repeat_interleave(self.clone_size, dim=0)

        # mask the input
        x = torch.gather(x, dim=1, index=ids_keep)  # B * clone_size, L * (1 - mask_ratio), D
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)

        # separate cls from the rest
        cls_tokens = x[:, 0]
        patch_tokens = x[:, 1:]
        
        return cls_tokens, patch_tokens, mask, ids_restore

    def teacher_forward(self, x, mask_ratio=None):
        x = self.patch_embed(x)  # B, C=1, T=512, F=128 -> B, L=256, D=768
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        features = []
        for blk in self.blocks:
            x, t = blk(x)
            features.append(t[:, 1:, :].clone())
        cls_tokens = t[:, 0]  # last layer cls token
        return cls_tokens, features
       
    def forward(self, x, mask_ratio=None): 
        return self.forward_fn(x, mask_ratio)


# Student model for EAT
class EAT_Student(nn.Module):
    
    def __init__(self,
                 input_shape=(512, 128), 
                 patch_size=(16, 16),
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path_rate=0,
                 pos_trainable=False,
                 clone_size=16,
                 mask_mode='inv',
                 cls_task='regression',
                 dinohead_kwargs={'out_dim': 65536, 'use_bn': False, 'nlayers': 3, 'hidden_dim': 2048, 'bottleneck_dim': 256, 'mlp_bias': True},
                 decoder_cls=CNN2dDecoder,
                 decoder_kwargs={'kernel_size': 3, 'stride': 1, 'padding': 'same', 'groups': 16, 'activation': nn.GELU, 'add_residual': True, 'num_layers': 6},
                ):
        
        super().__init__()
        assert cls_task in ['regression', 'clustering']
        self.cls_task = cls_task
        self.clone_size = clone_size
        self.encoder = ViT_MaskedEncoder(input_shape, patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path_rate, pos_trainable, clone_size, mode='student', mask_mode=mask_mode)
        if cls_task == 'clustering':
            dinohead_kwargs['in_dim'] = embed_dim
            self.head = DINOHead(**dinohead_kwargs)
        decoder_kwargs['num_freq_patches'] = self.encoder.patch_embed.patch_ft[0]
        decoder_kwargs['num_time_patches'] = self.encoder.patch_embed.patch_ft[1]
        decoder_kwargs['dim'] = embed_dim
        self.decoder = decoder_cls(**decoder_kwargs)
        self.initialize_weights()
        
        
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed_flexible(self.encoder.pos_embed.shape[-1], self.encoder.patch_embed.patch_ft, cls_token=True)
        self.encoder.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.encoder.cls_token, std=.02)
        torch.nn.init.normal_(self.decoder.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.LayerNorm):
            # nn.init.constant_(m.bias, 0)
            # nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask_ratio=0.8):
        """
        args:
            x - input mel-spectrogram of shape B, 1, T, F 
        """
        cls_tokens, patch_tokens, mask, ids_restore = self.encoder(x, mask_ratio)
        patch_tokens = self.decoder(patch_tokens, ids_restore)
        return cls_tokens, patch_tokens, mask

    
# Teacher model for EAT
class EAT_Teacher(nn.Module):
    
    def __init__(self,
                 input_shape=(512, 128),
                 patch_size=(16, 16),
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop=0,
                 attn_drop=0.,
                 drop_path_rate=0,
                 pos_trainable=False,
                 clone_size=16,
                 cls_task='regression',
                 dinohead_kwargs={'out_dim': 65536, 'use_bn': False, 'nlayers': 3, 'hidden_dim': 2048, 'bottleneck_dim': 256, 'mlp_bias': True},
                 average_top_k_layers=12,
                 instance_norm_target_layer=True,  # based on EAT config
                 batch_norm_target_layer=False,  # based on EAT config
                 layer_norm_target_layer=False,  # based on EAT config
                 layer_norm_targets=True,  # based on EAT config
                 instance_norm_targets=False,  # based on EAT config
                ):
        
        super().__init__()
        assert cls_task in ['regression', 'clustering']
        self.cls_task = cls_task
        self.clone_size = clone_size
        self.encoder = ViT_MaskedEncoder(input_shape, patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path_rate, pos_trainable, clone_size, mode='teacher')
        if cls_task == 'clustering':
            dinohead_kwargs['in_dim'] = embed_dim
            self.head = DINOHead(**dinohead_kwargs)
        self.average_top_k_layers = average_top_k_layers
        self.instance_norm_target_layer = instance_norm_target_layer
        self.batch_norm_target_layer = batch_norm_target_layer
        self.layer_norm_target_layer = layer_norm_target_layer
        self.layer_norm_targets = layer_norm_targets
        self.instance_norm_targets = instance_norm_targets

    
    def make_targets(self, y):   
        y = y[-self.average_top_k_layers:]
        permuted = False
        if self.instance_norm_target_layer or self.batch_norm_target_layer:  # BTC -> BCT
            y = [tl.transpose(1, 2) for tl in y]
            permuted = True
        if self.batch_norm_target_layer:
            y = [F.batch_norm(tl.float(), running_mean=None, running_var=None, training=True) for tl in y]
        if self.instance_norm_target_layer:
            y = [F.instance_norm(tl.float()) for tl in y]
        if permuted: # BCT -> BTC
            y = [tl.transpose(1, 2) for tl in y]
        if self.layer_norm_target_layer:
            y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
    
        y = torch.stack(y).mean(dim=0)  # average layers' outputs
        if self.layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])
        if self.instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)
        return y

    def forward(self, x):
        """
        args:
            x - input mel-spectrogram of shape B, 1, T, F 
        """
        cls_tokens, patch_tokens = self.encoder(x, 0)  # outputs final cls_token and a list of all transformer layers' embeddings (e.g., 12 layer in base vit)
        patch_tokens = self.make_targets(patch_tokens)
        patch_tokens = patch_tokens.repeat_interleave(self.clone_size, dim=0)
        cls_tokens = cls_tokens.repeat_interleave(self.clone_size, dim=0)
        return cls_tokens, patch_tokens


# Clustering Modules
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256, mlp_bias=True):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = self._build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        # I do not know if doing this is necessary or not, they did it, so let's keep it.
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        # eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-8)
        x = self.last_layer(x)
        return x

    def _build_mlp(self, nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
        if nlayers == 1:
            return nn.Linear(in_dim, bottleneck_dim, bias=bias)
        else:
            layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
            return nn.Sequential(*layers)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.len_teacher_output = None
        self.batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp=0.07):
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp=0.07, n_iterations=3):
        teacher_output = teacher_output.float()
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def gini_impurity(self, p):
        """
        p - teacher softmax predictions (B, D)
        """
        return (1.0 - (p ** 2).sum(dim=1)).mean()

    def forward(self, student_logits, teacher_probs, student_temp=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        if student_temp is None:
            student_temp = self.student_temp
        lsm = F.log_softmax(student_logits / student_temp, dim=-1)
        return -(teacher_probs * lsm).sum(dim=-1).mean()

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        
    @torch.no_grad()
    def apply_center_update(self):
        if not self.updated:
            _t = self.batch_center / (self.len_teacher_output)
            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)
            self.updated = True


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.autocast(device_type=student_output.device.type, enabled=False): #change
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss


def var_loss(x):
    """
    x - student logits (B, D)
    """
    x_centered = x - x.mean(dim=0)
    s = torch.sqrt(x_centered.var(dim=0) + 0.0001)
    return F.relu(1. - s).mean()
  

# Utils
class RiseRunDecay(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, steps_in_epoch=None, warmup=10, constant=0, total_epochs=None, lowest_lr=1e-6):

        self.warmup = warmup * steps_in_epoch
        self.constant = self.warmup + (constant * steps_in_epoch)
        self.final_step = total_epochs * steps_in_epoch
        self.decay_interval = self.final_step - self.constant
        self.lowest_lr = lowest_lr
        super().__init__(optimizer)

    def get_lr(self):
        current_iteration = self.last_epoch
        if current_iteration <= self.warmup:
            factor = current_iteration / self.warmup
        elif current_iteration <= self.constant:
            factor = 1.0
        else:
            current_iteration = self.last_epoch - self.constant
            factor = 0.5 * (1 + math.cos(math.pi * current_iteration / self.decay_interval))

        return [lr * factor if (lr * factor) > self.lowest_lr else self.lowest_lr for lr in self.base_lrs]
      

class EMA_Weight_Decay_Scheduler:
    def __init__(self, decay_start=0.9998, decay_end=0.99999, max_iter=None):
        self.decays = np.linspace(decay_start, decay_end, max_iter, dtype=np.float32).tolist() + [decay_end]
        self.max_iter = max_iter
        self.counter = 0
        
    def step(self):
        w = self.decays[self.counter]
        self.counter = min(self.counter + 1, self.max_iter)
        return w
        
        
class Sigmoid_Rampup_Scheduler:
    def __init__(self, scale=-5.0, max_iter=None):
        self.scale = scale
        self.max_iter = max_iter
        self.counter = 0
        
    def step(self):
        phase_square = (1.0 - self.counter / self.max_iter) ** 2
        self.counter = min(self.counter + 1, self.max_iter)
        return math.exp(self.scale * phase_square)


@torch.no_grad()
def ema_update(ema_model, model, buffers=True, decay=.999):
    for p_avg, p in zip(ema_model.parameters(), model.parameters()):
        p_avg.data = decay * p_avg.data + (1. - decay) * p.data
    if buffers:
        for (n, b_avg), (n2, b) in zip(ema_model.named_buffers(), model.named_buffers()):
            if n.split('.')[-1] == 'num_batches_tracked':
                b_avg.data = b.data
            else:
                b_avg.data = decay * b_avg.data + (1. - decay) * b.data


def masked_reconstruction_loss(pred, target, mask):
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)
    return (loss * mask).sum() / mask.sum()


def load_eat_audioset_pretrained_state(vit_encoder, audioset_eat_state_path='EAT-base_epoch30_pt.pt'):
    audioset_state = torch.load(audioset_eat_state_path)
    model_state = vit_encoder.state_dict()
    model_state['cls_token'] = audioset_state['model']['modality_encoders.IMAGE.extra_tokens'].clone()
    model_state['patch_embed.proj.weight'] = audioset_state['model']['modality_encoders.IMAGE.local_encoder.proj.weight'].clone()
    model_state['patch_embed.proj.bias'] = audioset_state['model']['modality_encoders.IMAGE.local_encoder.proj.bias'].clone()
    # model_state['pos_embed'] = audioset_state['model']['modality_encoders.IMAGE.fixed_positional_encoder.positions'][:, :257].clone()
    model_state['norm.weight'] = audioset_state['model']['modality_encoders.IMAGE.context_encoder.norm.weight'].clone()
    model_state['norm.bias'] = audioset_state['model']['modality_encoders.IMAGE.context_encoder.norm.bias'].clone()
    for k in audioset_state['model'].keys():
        if k[:6] == 'blocks':
            model_state[k] = audioset_state['model'][k].clone()
    _ = vit_encoder.load_state_dict(model_state, strict=False)
    print(_)
    return vit_encoder