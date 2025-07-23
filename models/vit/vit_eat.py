from functools import partial
import torch
from torch import nn
from util.pos_embed import get_2d_sincos_pos_embed_flexible
import hydra
import lightning as L
from torchmetrics import MetricCollection
from util.lr_decay import param_groups_lrd

from ..components.attentive_pooling import AttentivePooling
from ..components.cosine_warmup import CosineWarmupScheduler

class VIT_EAT(L.LightningModule):
    def __init__(self, 
                 img_size_x,
                 img_size_y,
                 patch_size,
                 in_chans,
                 embed_dim,
                 global_pool,
                 norm_layer,
                 mlp_ratio,
                 qkv_bias,
                 eps,
                 drop_path,
                 num_heads,
                 depth,
                 num_classes,
                 optimizer,
                 scheduler,
                 pretrained_weights_path, 
                 target_length,
                 loss,
                 metric_cfg,
                 mask_t_prob,
                 mask_f_prob,
                 mask2d,
                 mask_mode='rand',
                 pos_trainable=False,
                 ppnet_cfg=None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.img_size = (img_size_x, img_size_y)
        self.patch_size = (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.qkv_bias = qkv_bias
        self.global_pool = global_pool
        self.mask_mode = mask_mode
        self.pos_trainable = pos_trainable
        
        self.mask_2d = mask2d
        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob
        
        self.pretrained_weights_path = pretrained_weights_path
        self.target_length = target_length
        self.ppnet_cfg = ppnet_cfg

        # build model components
        self._build_model(eps, drop_path)
        
        self.loss = hydra.utils.instantiate(loss)
        self.optimizer = None
        self.optimizer_cfg = optimizer.target
        self.train_batch_size = optimizer.extras.train_batch_size
        self.layer_decay = optimizer.extras.layer_decay
        self.decay_type = optimizer.extras.decay_type
        self.scheduler_cfg = scheduler
        
        # setup metrics
        metric = hydra.utils.instantiate(metric_cfg)
        additional_metrics = []
        if metric_cfg.get("additional"):
            for _, metric_cfg in metric_cfg.additional.items():
                additional_metrics.append(hydra.utils.instantiate(metric_cfg))
        add_metrics = MetricCollection(additional_metrics)
        
        self.train_metric = metric.clone()
        self.val_metric = metric.clone()
        self.test_metric = metric.clone()
        self.test_add_metrics = add_metrics.clone()
        self.val_add_metrics = add_metrics.clone()
        
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
        
  
    def _build_model(self, eps, drop_path_rate):
        """Build the model components"""
        # patch embedding
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, 1, self.embed_dim)
        
        # positional embedding
        pos_embed = get_2d_sincos_pos_embed_flexible(
            self.embed_dim, self.patch_embed.patch_ft, cls_token=True)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0), 
            requires_grad=self.pos_trainable)
        
        # cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        
        # norm layer
        norm_layer = partial(nn.LayerNorm, eps=eps)
        self.fc_norm = norm_layer(self.embed_dim)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([
            AltBlock(
                self.embed_dim, self.num_heads, self.mlp_ratio, 
                self.qkv_bias, 0., 0., dpr[i], norm_layer=norm_layer
            ) for i in range(self.depth)
        ])
        
        self.norm = norm_layer(self.embed_dim)

        if self.ppnet_cfg:
            # ppnet head
            from ..ppnet.ppnet import PPNet

            h_patches = self.img_size[1] // self.patch_size[0]  # freq patches (8 for 128/16)
            w_patches = self.img_size[0] // self.patch_size[1]  # time patches (32 for 512/16)

            self.ppnet = PPNet(
                num_prototypes=self.ppnet_cfg.num_prototypes,
                channels_prototypes=self.ppnet_cfg.channels_prototypes,
                h_prototypes=self.ppnet_cfg.h_prototypes,
                w_prototypes=self.ppnet_cfg.w_prototypes,
                num_classes=self.ppnet_cfg.num_classes,
                topk_k=self.ppnet_cfg.topk_k,
                margin=self.ppnet_cfg.margin,
                init_weights=self.ppnet_cfg.init_weights,
                add_on_layers_type=self.ppnet_cfg.add_on_layers_type,
                incorrect_class_connection=self.ppnet_cfg.incorrect_class_connection,
                correct_class_connection=self.ppnet_cfg.correct_class_connection,
                bias_last_layer=self.ppnet_cfg.bias_last_layer,
                non_negative_last_layer=self.ppnet_cfg.non_negative_last_layer,
                embedded_spectrogram_height=self.ppnet_cfg.embedded_spectrogram_height,
            )
        else:
            # normal classification head
            self.head = nn.Linear(self.embed_dim, self.num_classes)
        
        # attentive pooling
        if self.global_pool == "attentive":
            self.attentive_probe = AttentivePooling(dim=self.embed_dim, num_heads=self.num_heads)
        
        # masking augs
        if self.mask_mode == 'rand':
            self.mask_fn = self.random_masking
        
        elif self.mask_mode == 'inv':
            self.mask_fn = self.inverse_block_mask
        else:
            self.mask_fn = None

    def random_masking(self, shape, mask_ratio, *args):
        """Random masking as in original EAT"""
        if mask_ratio == 0:
            return None, None, None
        
        B, L, D = shape
        len_keep = int(L * (1 - mask_ratio))    
        noise = torch.rand(B, L, device=self.device)
        
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        mask = torch.ones([B, L], device=self.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        ids_restore = ids_restore.unsqueeze(-1).repeat(1, 1, D)
        ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        
        return mask, ids_keep, ids_restore

    def inverse_block_mask(self, shape, mask_ratio=0.2, num_freq_patches=8, num_time_patches=32, mask_length=5, mask_prob_adjust=0.07, require_same_masks=True):
        """Inverse block masking as in original EAT"""
        if mask_ratio == 0:
            return None, None, None
        
        assert mask_length > 1
        
        B, L, D = shape
        d = (num_time_patches, num_freq_patches)
        mask_ratio = 1 - mask_ratio    
        
        mask = torch.zeros((B, d[0], d[1]), device=self.device)
        masking_size = int(L * ((mask_ratio + mask_prob_adjust) / (mask_length ** 2)))
        mask_inds = torch.randint(0, L, size=(B, masking_size), device=self.device)
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
                if n > target_len:
                    to_unmask = torch.multinomial(m, int(n - target_len), replacement=False)
                    m[to_unmask] = 0
                elif n < target_len:
                    to_mask = torch.multinomial((1 - m), int(target_len - n), replacement=False)
                    m[to_mask] = 1
                    
        mask = 1 - mask  
        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)
        len_keep = L - mask[0].sum()
        ids_keep = ids_shuffle[:, :len_keep]
        ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        return mask.float(), ids_keep, ids_restore

    def random_masking_2d(self, x):
        """2D masking for time and frequency dimensions"""
        N, L, D = x.shape
        T = self.img_size[0] // self.patch_size[0]  # time patches
        F = self.img_size[1] // self.patch_size[1]  # freq patches

        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - self.mask_t_prob))
        noise = torch.rand(N, T, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        x = torch.gather(x, dim=1, index=index)

        x = x.permute(0,2,1,3)  # N T' F D => N F T' D
        len_keep_F = int(F * (1 - self.mask_f_prob))
        noise = torch.rand(N, F, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep_F]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0,2,1,3)  # N F' T' D => N T' F' D 
        x_masked = x_masked.reshape(N, len_keep_F*len_keep_T, D)
            
        return x_masked, None, None

    def forward_features(self, x, mask_ratio=0.0):
        """Forward pass through the model features"""
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # apply masking 
        if mask_ratio > 0 and self.mask_fn is not None:
            num_freq_patches, num_time_patches = self.patch_embed.patch_ft
            mask, ids_keep, ids_restore = self.mask_fn(
                x.shape, mask_ratio, num_freq_patches, num_time_patches)
            if mask is not None:
                x = torch.gather(x, dim=1, index=ids_keep)
        
        # apply 2D masking 
        if self.mask_2d and (self.mask_t_prob > 0.0 or self.mask_f_prob > 0.0):
            x, _, _ = self.random_masking_2d(x)
        
        # add class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # transformer blocks
        for blk in self.blocks:
            x, _ = blk(x)
        
        if self.ppnet_cfg:
            # PPNet head processing
            x_cls = x[:, 0, :]  # cls token
            x_patch = x[:, 1:, :]  # patch tokens
            
            if self.ppnet_cfg.focal_similarity:
                # focal similarity: subtract cls from patches
                z_f = x_patch - x_cls.unsqueeze(1)
                # reshape to spatial format for PPNet
                h_patches = self.img_size[1] // self.patch_size[0]  # freq patches
                w_patches = self.img_size[0] // self.patch_size[1]  # time patches
                x = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, h_patches, w_patches)
            else:
                # regular patch processing
                h_patches = self.img_size[1] // self.patch_size[0]
                w_patches = self.img_size[0] // self.patch_size[1]
                x = x_patch.permute(0, 2, 1).reshape(B, self.embed_dim, h_patches, w_patches)
            
            logits, _ = self.ppnet(x)
            return logits
        else:
            if self.global_pool == "average": 
                x = x[:, 1:, :].mean(dim=1)
                outcome = self.fc_norm(x)
            elif self.global_pool == "attentive":
                outcome = self.attentive_probe(x)
                outcome = self.fc_norm(outcome)
            elif self.global_pool == "cls":
                x = self.norm(x)
                outcome = x[:, 0]
            else:
                x = self.norm(x)
                outcome = x[:, 0]
            
        return outcome

    def forward(self, x, mask_ratio=0.0):
        x = self.forward_features(x, mask_ratio)
        if self.ppnet_cfg:
            pred = x
        else:
            pred = self.head(x)
        return pred

    def training_step(self, batch, batch_idx):
        """Training step"""
        audio = batch["audio"]
        targets = batch["label"]
        
        if self.ppnet_cfg:
            logits = self(audio)
            targets = targets.long()
            
            # PPNet loss
            try:
                bce_loss = self.loss(logits, targets.float())
            except:
                bce_loss = self.loss(logits, targets)
            
            orthogonality_loss = self._calculate_orthogonality_loss()
            loss = bce_loss + orthogonality_loss
            
            self.log('bce_loss', bce_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('orthogonality_loss', orthogonality_loss, on_step=True, on_epoch=True, prog_bar=True)
        else:
            # normal classification
            pred = self(audio)
            targets = targets.long()
            
            try:
                loss = self.loss(pred, targets)
            except:
                loss = self.loss(pred, targets.float())
            
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        pred = self(audio)
        targets = targets.long()
        
        try:
            loss = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())

        self.val_predictions.append(pred.detach().cpu())
        self.val_targets.append(targets.detach().cpu())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions)
        targets = torch.cat(self.val_targets)
        metric = self.val_metric(preds, targets)
        self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)

        self.val_add_metrics(preds, targets)
        for name, metric in self.val_add_metrics.items():
            self.log(f'valid_{name}', metric, on_epoch=True, prog_bar=True)

        self.val_predictions = []
        self.val_targets = []
    
    def test_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        # disable masking for test
        pred = self(audio, mask_ratio=0.0)

        targets = targets.long()
        try:
            loss = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())
        
        self.test_predictions.append(pred.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    
    def on_test_epoch_end(self):
        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)
        self.test_metric(preds, targets)
        self.log(f'test_{self.test_metric.__class__.__name__}', self.test_metric, on_epoch=True, prog_bar=True)

        self.test_add_metrics(preds, targets)
        for name, metric in self.test_add_metrics.items():
            self.log(f'test_{name}', metric, on_epoch=True, prog_bar=True)

        self.test_predictions = []
        self.test_targets = []

    def configure_optimizers(self):       
        if self.ppnet_cfg:
            # PPNet-specific optimizer
            from util.lr_decay import param_groups_lrd_pp
            
            
            if self.layer_decay:
                params = param_groups_lrd_pp(
                    model=self,
                    weight_decay=self.optimizer_cfg["weight_decay"],
                    no_weight_decay_list=self.no_weight_decay(),
                    layer_decay=self.layer_decay,
                    decay_type=self.decay_type,
                    last_layer_lr=self.ppnet_cfg.last_layer_lr,
                    prototype_lr=self.ppnet_cfg.prototype_lr,
                )
                
                self.optimizer = hydra.utils.instantiate(self.optimizer_cfg, params)
            else:
                # fallback for PPNet without layer decay
                params = self.parameters()
                self.optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=params)
                
        else:
            # normal optimizer for standard classification head
            if self.layer_decay:
                params = param_groups_lrd(
                    model=self,
                    weight_decay=self.optimizer_cfg["weight_decay"],
                    no_weight_decay_list=self.no_weight_decay(),
                    layer_decay=self.layer_decay,
                    decay_type=self.decay_type
                )
                self.optimizer = hydra.utils.instantiate(self.optimizer_cfg, params)
            else:
                self.optimizer = hydra.utils.instantiate(
                    self.optimizer_cfg, params=self.parameters())

        # scheduler configuration (same for both PPNet and normal)
        if self.scheduler_cfg: 
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.067
            num_warmup_steps = num_training_steps * warmup_ratio

            scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=num_warmup_steps,
                total_steps=num_training_steps
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  
                "frequency": 1,
                "name": "lr_cosine"
            }
            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}
        
        return {"optimizer": self.optimizer}

    def no_weight_decay(self):
        """Parameters that should not have weight decay"""
        return {'pos_embed', 'cls_token'}

    def load_pretrained_weights(self, pretrained_weights_path, dataset_name):
        print(f"Loading EAT pretrained weights from {pretrained_weights_path}")
        
        try:
            checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
            
            # handle different checkpoint formats
            if "encoder" in checkpoint:
                # direct EAT encoder checkpoint
                encoder_state = checkpoint["encoder"]
                pretrained_state_dict = {}
                
                # map encoder weights to ViT structure
                for key, value in encoder_state.items():
                    if key.startswith("patch_embed."):
                        pretrained_state_dict[key] = value
                    elif key.startswith("pos_embed"):
                        pretrained_state_dict[key] = value
                    elif key.startswith("cls_token"):
                        pretrained_state_dict[key] = value
                    elif key.startswith("blocks."):
                        pretrained_state_dict[key] = value
                    elif key.startswith("norm."):
                        pretrained_state_dict[key] = value
                        
            elif "state_dict" in checkpoint:
                # lightning checkpoint format
                state_dict = checkpoint["state_dict"]
                pretrained_state_dict = {}
                
                for key, value in state_dict.items():
                    if key.startswith("student.encoder."):
                        new_key = key.replace("student.encoder.", "")
                        
                        # map EAT naming to VIT_EAT naming
                        new_key = self._map_eat_to_vit_key(new_key)
                        pretrained_state_dict[new_key] = value
                    elif key.startswith("encoder."):
                        new_key = key.replace("encoder.", "")
                        new_key = self._map_eat_to_vit_key(new_key)
                        pretrained_state_dict[new_key] = value
                        
            elif "model" in checkpoint:
                # check if this is an original EAT checkpoint with modality encoders
                if any(key.startswith("modality_encoders.IMAGE.") for key in checkpoint["model"].keys()):
                    # original EAT AudioSet pretrained checkpoint
                    audioset_state = checkpoint["model"]
                    pretrained_state_dict = {}
                    
                    # map original EAT keys to VIT_EAT keys
                    if "modality_encoders.IMAGE.extra_tokens" in audioset_state:
                        pretrained_state_dict["cls_token"] = audioset_state["modality_encoders.IMAGE.extra_tokens"]
                    
                    if "modality_encoders.IMAGE.local_encoder.proj.weight" in audioset_state:
                        pretrained_state_dict["patch_embed.proj.weight"] = audioset_state["modality_encoders.IMAGE.local_encoder.proj.weight"]
                    
                    if "modality_encoders.IMAGE.local_encoder.proj.bias" in audioset_state:
                        pretrained_state_dict["patch_embed.proj.bias"] = audioset_state["modality_encoders.IMAGE.local_encoder.proj.bias"]
                    
                    if "modality_encoders.IMAGE.fixed_positional_encoder.positions" in audioset_state:
                        # truncate to first 257 positions (cls + patches)
                        pretrained_state_dict["pos_embed"] = audioset_state["modality_encoders.IMAGE.fixed_positional_encoder.positions"][:, :257]
                    
                    if "modality_encoders.IMAGE.context_encoder.norm.weight" in audioset_state:
                        pretrained_state_dict["norm.weight"] = audioset_state["modality_encoders.IMAGE.context_encoder.norm.weight"]
                    
                    if "modality_encoders.IMAGE.context_encoder.norm.bias" in audioset_state:
                        pretrained_state_dict["norm.bias"] = audioset_state["modality_encoders.IMAGE.context_encoder.norm.bias"]
                    
                    # copy all transformer blocks 
                    for key, value in audioset_state.items():
                        if key.startswith("blocks."):
                            pretrained_state_dict[key] = value
                            
                    print(f"Loaded original EAT AudioSet checkpoint with {len(pretrained_state_dict)} parameters")
                    
                else:
                    # standard model checkpoint
                    state_dict = checkpoint["model"]
                    pretrained_state_dict = {}
                    
                    for key, value in state_dict.items():
                        if key.startswith("student.encoder."):
                            new_key = key.replace("student.encoder.", "")
                            new_key = self._map_eat_to_vit_key(new_key)
                            pretrained_state_dict[new_key] = value
                        elif key.startswith("encoder."):
                            new_key = key.replace("encoder.", "")
                            new_key = self._map_eat_to_vit_key(new_key)
                            pretrained_state_dict[new_key] = value
                        else:
                            pretrained_state_dict[key] = value
            else:
                pretrained_state_dict = checkpoint
            
            # remove head weights if num_classes doesn't match
            current_state = self.state_dict()
            for k in ['head.weight', 'head.bias', 'fc.weight', 'fc.bias']:
                if k in pretrained_state_dict and k in current_state:
                    if pretrained_state_dict[k].shape != current_state[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint due to shape mismatch")
                        del pretrained_state_dict[k]
            
            # Lload the state dict
            missing_keys, unexpected_keys = self.load_state_dict(pretrained_state_dict, strict=False)
            
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
            
            # reinitialize positional embeddings if needed (e.g. AudioSet)
            if self.target_length != 512:
                print(f"Reinitializing positional embeddings for target length {self.target_length}")
                patch_hw = (self.img_size[1] // self.patch_size, self.img_size[0] // self.patch_size)
                pos_embed = get_2d_sincos_pos_embed_flexible(
                    self.pos_embed.size(-1), patch_hw, cls_token=True)
                self.pos_embed.data = torch.from_numpy(pos_embed).float().unsqueeze(0)
                                
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Continuing with random initialization...")

    def _map_eat_to_vit_key(self, key):
        """Map EAT encoder key names to VIT_EAT key names"""
        # handle layer norm naming: ln1/ln2 -> norm1/norm2
        if ".ln1." in key:
            key = key.replace(".ln1.", ".norm1.")
        elif ".ln2." in key:
            key = key.replace(".ln2.", ".norm2.")
        
        # handle MLP naming: ff.ffn.0 -> mlp.fc1, ff.ffn.3 -> mlp.fc2
        if ".ff.ffn.0." in key:
            key = key.replace(".ff.ffn.0.", ".mlp.fc1.")
        elif ".ff.ffn.3." in key:
            key = key.replace(".ff.ffn.3.", ".mlp.fc2.")
        
        return key

    
    def _calculate_orthogonality_loss(self) -> torch.Tensor:
        orthogonalities = self.ppnet.get_prototype_orthogonalities()
        orthogonality_loss = torch.norm(orthogonalities)

        # normalize the orthogonality loss
        normalized_orthogonality_loss = orthogonality_loss / orthogonalities.numel()

        return normalized_orthogonality_loss

# Vit Encoder for original EAT with AudioSet pretraining
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


class ViT(nn.Module):
    def __init__(self, num_classes=None, input_shape=(512, 128), patch_size=(16, 16), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop=0., attn_drop=0., drop_path_rate=0., pos_trainable=False, mask_mode='rand'):
        super().__init__()
        assert mask_mode in ['rand', 'inv']
        if mask_mode == 'rand':
            self.mask_fn = self.random_masking
        else:
            self.mask_fn = self.inverse_block_mask
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(input_shape, patch_size, 1, embed_dim)
        pos_embed = get_2d_sincos_pos_embed_flexible(embed_dim, self.patch_embed.patch_ft, cls_token=True)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=pos_trainable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02) # 1, 1, 768
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([AltBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    # def resize_pos_emedding(self, new_size=5000):
        # embed_dim = self.pos_embed.shape[-1]
        # self.register_buffer("pos_embed", get_positional_encoding(embed_dim, max_len=new_size))
        
    def inverse_block_mask(self, shape, mask_ratio=0.2, num_freq_patches=8, num_time_patches=32, mask_length=5, mask_prob_adjust=0.07, require_same_masks=True):
        
        if mask_ratio == 0:
            return None, None, None
        
        assert mask_length > 1
        
        B, L, D = shape
        # B = B * self.clone_size
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
    
    def random_masking(self, shape, mask_ratio, *args):
        
        if mask_ratio == 0:
            return None, None, None
        
        B, L, D = shape  # batch, length, dim
        # B *= self.clone_size
        
        len_keep = int(L * (1 - mask_ratio))    
        noise = torch.rand(B, L)  # noise in [0, 1]
        
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
    
    def forward(self, x, mask_ratio=0.2):
        x = self.patch_embed(x)  # B, 1, T, F -> B, L=256, D=768
        x = x + self.pos_embed[:, 1:, :]
        num_freq_patches, num_time_patches = self.patch_embed.patch_ft
        mask, ids_keep, ids_restore = self.mask_fn(x.shape, mask_ratio, num_freq_patches, num_time_patches)
        if mask is not None:
            x = torch.gather(x, dim=1, index=ids_keep.to(x.device))  # B, L * (1 - mask_ratio), D

        # add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # encode
        for blk in self.blocks:
            x, t = blk(x)
        x = self.norm(x)

        # predict
        cls_tokens = x[:, 0]
        return self.fc(cls_tokens)


# # Utils
# class RiseRunDecay(torch.optim.lr_scheduler._LRScheduler):

#     def __init__(self, optimizer, steps_in_epoch=None, warmup=10, constant=0, total_epochs=None, lowest_lr=1e-6):

#         self.warmup = warmup * steps_in_epoch
#         self.constant = self.warmup + (constant * steps_in_epoch)
#         self.final_step = total_epochs * steps_in_epoch
#         self.decay_interval = self.final_step - self.constant
#         self.lowest_lr = lowest_lr
#         super().__init__(optimizer)

#     def get_lr(self):
#         current_iteration = self.last_epoch
#         if current_iteration <= self.warmup:
#             factor = current_iteration / self.warmup
#         elif current_iteration <= self.constant:
#             factor = 1.0
#         else:
#             current_iteration = self.last_epoch - self.constant
#             factor = 0.5 * (1 + math.cos(math.pi * current_iteration / self.decay_interval))

#         return [lr * factor if (lr * factor) > self.lowest_lr else self.lowest_lr for lr in self.base_lrs]


# class EMA_Weight_Decay_Scheduler:
#     def __init__(self, decay_start=0.9998, decay_end=0.99999, max_iter=None):
#         self.decays = np.linspace(decay_start, decay_end, max_iter, dtype=np.float32).tolist() + [decay_end]
#         self.max_iter = max_iter
#         self.counter = 0
        
#     def step(self):
#         w = self.decays[self.counter]
#         self.counter += 1
#         self.counter = min(self.counter, self.max_iter)
#         return w


# @torch.no_grad()
# def ema_update(ema_model, model, buffers=True, decay=.999):
#     for p_avg, p in zip(ema_model.parameters(), model.parameters()):
#         p_avg.data = decay * p_avg.data + (1. - decay) * p.data
#     if buffers:
#         for (n, b_avg), (n2, b) in zip(ema_model.named_buffers(), model.named_buffers()):
#             if n.split('.')[-1] == 'num_batches_tracked':
#                 b_avg.data = b.data
#             else:
#                 b_avg.data = decay * b_avg.data + (1. - decay) * b.data


# def train_step(model, optimizer, train_loader, ema_model=None, scheduler=None, device='cuda', clip_norm=4., mask_ratio=0.2, ema_scheduler=None):
#     losses = []
#     model.train()
#     for x, y in tqdm(train_loader, leave=False):
#         x, y = x.to(device), y.to(device)  # x: B, 1, T=512, F=128 ; y: B, C=num_classes
#         with torch.autocast(device_type=device, dtype=torch.bfloat16):
#             logits = model(x, mask_ratio)
#         logits = logits.float()
#         loss = F.binary_cross_entropy_with_logits(logits, y)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
#         optimizer.step()
#         if scheduler is not None:
#             scheduler.step()
#         optimizer.zero_grad()
#         losses.append(loss.detach().cpu().item())
#         if ema_model is not None:
#             if ema_scheduler is not None:
#                 decay = ema_scheduler.step()
#             else:
#                 decay = 0.999
#             ema_update(ema_model, model, decay=decay)
#     return np.mean(losses)


# @torch.inference_mode()
# def val_step(model_, val_loader, device='cuda', prefix='val_'):
#     y_true, y_pred, loss = [], [], []
#     model_.eval()
#     for x, y in tqdm(val_loader, leave=False):
#         x, y = x.to(device), y.to(device)
#         with torch.autocast(device_type=device, dtype=torch.bfloat16):
#             logits = model_(x, 0)
#         logits = logits.float()
#         loss.append(F.binary_cross_entropy_with_logits(logits, y).cpu().item())
#         y_true.append(y.cpu())
#         y_pred.append(logits.sigmoid().cpu())
#     cache = get_metrics(torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(), prefix)
#     cache[prefix+'loss'] = np.mean(loss)
#     return cache


# def get_metrics(y_true, y_pred, prefix=''):
    
#     y_pred_hard = y_pred.round()
#     p, r, f, _ = metrics.precision_recall_fscore_support(y_true, y_pred_hard, average='macro', zero_division=0)
#     cache = {prefix+'f1_macro': f * 100,
#              prefix+'precision': p * 100,
#              prefix+'recall': r * 100,
#              prefix+'f1_micro': metrics.f1_score(y_true, y_pred_hard, average='micro', zero_division=0) * 100,
#              prefix+'auc': metrics.roc_auc_score(y_true, y_pred, average='macro') * 100,
#              prefix+'mAP': metrics.average_precision_score(y_true, y_pred, average='macro') * 100,
#              }
#     post_pos, post_neg = get_avg_posterior(y_true, y_pred)
#     cache[prefix+'post_pos_mu'] = post_pos['mu'] * 100
#     cache[prefix+'post_pos_std'] = post_pos['std'] * 100
#     cache[prefix+'post_neg_mu'] = post_neg['mu'] * 100
#     cache[prefix+'post_neg_std'] = post_neg['std'] * 100

#     # TODO (@Lukas):  please check if the top1 accuracy is correct here. 
#     # top1 accuracy: if the top predicted class is within the true labels
#     y_true, y_pred = torch.from_numpy(y_true), torch.from_numpy(y_pred)
#     mask = y_true.sum(dim=1) != 0
#     # mask_no_call = ~mask
#     # y_true_no_call = y_true[mask_no_call]
#     # y_pred_no_call = y_pred[mask_no_call]  
#     y_true = y_true[mask]
#     y_pred = y_pred[mask]
#     top1_index = y_pred.argmax(dim=1)
#     cache[prefix+'T1A'] = ((y_true[torch.arange(y_true.shape[0]), top1_index] == 1).float().sum() / y_true.shape[0]).item() * 100
#     return cache


# def get_avg_posterior(y_true, y_pred):
#     post_mus_pos, post_mus_neg = {}, {}
#     num_classes = y_pred.shape[1]
#     for k in range(num_classes):
#         y_true_k = y_true[:, k]
#         y_pred_k = y_pred[:, k]
#         post_mus_pos[k] = y_pred_k[y_true_k == 1].mean()
#         post_mus_neg[k] = 1 - y_pred_k[y_true_k == 0].mean()
#     post_mus_pos['mu'] = np.mean(list(post_mus_pos.values()))
#     post_mus_pos['std'] = np.std(list(post_mus_pos.values()))
#     post_mus_neg['mu'] = np.mean(list(post_mus_neg.values()))
#     post_mus_neg['std'] = np.std(list(post_mus_neg.values()))
#     return post_mus_pos, post_mus_neg

# def load_eat_audioset_pretrained_state(model, eat_state_path='EAT-base_epoch30_pt.pt'):
#     audioset_state = torch.load(eat_state_path)
#     model_state = model.state_dict()
#     model_state['cls_token'] = audioset_state['model']['modality_encoders.IMAGE.extra_tokens'].clone()
#     model_state['patch_embed.proj.weight'] = audioset_state['model']['modality_encoders.IMAGE.local_encoder.proj.weight'].clone()
#     model_state['patch_embed.proj.bias'] = audioset_state['model']['modality_encoders.IMAGE.local_encoder.proj.bias'].clone()
#     model_state['pos_embed'] = audioset_state['model']['modality_encoders.IMAGE.fixed_positional_encoder.positions'][:, :257].clone()
#     model_state['norm.weight'] = audioset_state['model']['modality_encoders.IMAGE.context_encoder.norm.weight'].clone()
#     model_state['norm.bias'] = audioset_state['model']['modality_encoders.IMAGE.context_encoder.norm.bias'].clone()
#     for k in audioset_state['model'].keys():
#         if k[:6] == 'blocks':
#             model_state[k] = audioset_state['model'][k].clone()
#     _ = model.load_state_dict(model_state, strict=False)
#     print(_)
#     return model





# # config
# epochs = 200
# device = 'cuda'
# model_type = 'vit_cnn2d'
# sample_dur = 5.11
# patch_size = (16, 16)
# use_ema_pretrained = False
# mask_mode = 'rand'
# mask_ratio = 0.2
# clip_norm = 4.
# compression = None  # pcen or other variants; I can add the code later if you want
# use_secondary_labels = False  # this helps me so much, doubling the metrics or even more sometimes!
# use_tf_mask = False
# use_bernoulli_noise = False  # this is better than tf-mask of SpecAugment in my experiments
# use_striped_mask = False  # not good
# use_patch_mask = False  # not good, this is similar to SSAST model masking in its pretraining. also, a bit similar to inverse-block masking, but quite the same.
# use_pitch_shift = False  # the squeeze here doesn't worth the juice!
# use_time_stretch = False  # not worth the troubles either, usually harmful in my experiments.
# use_bg_noise = True  # VOX no-call dataset
# use_color_noise = False
# use_random_gain = False
# use_mixup = True  # in my code, this mixup uses class-frequency weighted sampling, which helps a lot with unbalanced dataset.
# double_mixup = False  # a second mixup at the batch level, which helped in my code
# use_hrps = False  # harmonic-residual-percussive source separation and mixing again with random weights


# experiment_name = f"AudioSet_{model_type}_{mask_mode}({int(100*mask_ratio)})_{patch_size[0]}x{patch_size[1]}"
# if compression is not None: experiment_name += f"_{compression}"
# if use_tf_mask: experiment_name += "_tfmask"
# if use_bernoulli_noise: experiment_name += "_bern"
# if use_striped_mask: experiment_name += "_striped"
# if use_patch_mask: experiment_name += '_patchmask'
# if use_pitch_shift: experiment_name += "_pitch"
# if use_time_stretch: experiment_name += "_tstretch"
# if use_bg_noise: experiment_name += "_bgvox"
# if use_color_noise: experiment_name += "_colornoise"
# if use_random_gain: experiment_name += "_gain"
# if use_mixup: experiment_name += "_mixup"
# if double_mixup: experiment_name += "2"
# if use_hrps: experiment_name += "_hrps"
# if not use_secondary_labels: experiment_name += "_singlelabel"
# print(experiment_name)

# # saving and monitoring
# par_dir = Path(f'logs/')
# Path.mkdir(par_dir, parents=True, exist_ok=True)
# time_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# history_path = par_dir.joinpath(f'{experiment_name}_{int(sample_dur)}sec_{time_id}_history.csv')
# state_path = par_dir.joinpath(f'{experiment_name}_{int(sample_dur)}sec_{time_id}_state.pt')
# print(history_path)
# print(state_path)
# history = {'train_loss': [],
#            'ema_loss': [],
#            'ema_f1_macro': [],
#            'ema_precision': [],
#            'ema_recall': [],
#            'ema_f1_micro': [],
#            'ema_auc': [],
#            'ema_mAP': [],
#            'ema_post_pos_mu': [],
#            'ema_post_pos_std': [],
#            'ema_post_neg_mu': [],
#            'ema_post_neg_std': [],
#            'ema_T1A':[],
#            'test_loss': [],
#            'test_f1_macro': [],
#            'test_precision': [],
#            'test_recall': [],
#            'test_f1_micro': [],
#            'test_auc': [],
#            'test_mAP': [],
#            'test_post_pos_mu': [],
#            'test_post_pos_std': [],
#            'test_post_neg_mu': [],
#            'test_post_neg_std': [],
#          'test_T1A':[]}

# build models
# model = ViT(num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, dropout=0.1, drop_path_rate=0, mask_mode=mask_mode)
# ema_model = ViT(num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, dropout=0.1, drop_path_rate=0, mask_mode=mask_mode)
# model = load_eat_audioset_pretrained_state(model, eat_state_path='EAT-base_epoch30_pt.pt')
# ema_model.load_state_dict(model.state_dict())
# model.to(device).train()
# ema_model.to(device).eval()
# model.compile(mode='default')  # reduce-overhead
# ema_model.compile(mode='default')
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.05, betas=[0.9, 0.95])
# scheduler = RiseRunDecay(optimizer, steps_in_epoch=len(train_loader), warmup=20, total_epochs=epochs)
# ema_scheduler = EMA_Weight_Decay_Scheduler(decay_start=0.9998, decay_end=0.99999, max_iter=len(train_loader)*(epochs//4))
# print(f'#parameters: {sum(p.numel() for p in model.parameters()):_}')


# # run
# if __name__ == "__main__":
#     train_loader = None
#     pbar = tqdm(range(epochs), colour='green')
# for epoch in pbar:
#     train_loss = train_step(model, optimizer, train_loader, ema_model, scheduler, device, clip_norm, mask_ratio, ema_scheduler)
#     test_cache = val_step(model, test_loader, device, prefix='test_')
#     ema_test_cache = val_step(ema_model, test_loader, data_transforms, prefix='ema_')
#     history['train_loss'].append(train_loss)
#     # if val_loader is not None:  We don't have validation for now...
#     #     val_cache = val_step(model, val_loader, device, prefix='val_')
#     #     for k, v in val_cache.items():
#     #         history[k].append(v)
#     #     val_loss = val_cache['val_loss']
#     # else:
#     #     val_loss = -1
#     for k, v in test_cache.items():
#         history[k].append(v)
#     for k, v in ema_test_cache.items():
#         history[k].append(v)
#     pbar.set_description(f"loss={train_loss:.4f} test_loss={test_cache['test_loss']:.4f} test_mAP={test_cache['test_mAP']:.4f} test_f1={test_cache['test_f1_macro']:.4f}  ema_mAP={ema_test_cache['ema_mAP']:.4f} ema_f1={ema_test_cache['ema_f1_macro']:.4f}")