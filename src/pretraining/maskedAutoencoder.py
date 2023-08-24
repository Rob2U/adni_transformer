''' implementation of the Masked Autoencoder based on the https://github.com/facebookresearch/mae_st
    (license: https://raw.githubusercontent.com/facebookresearch/mae_st/27edea1326a8c0655ac0a339afed7effae2b8ba1/LICENSE)
    slight modifications where applied to the original code '''


import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from pretraining.pretraining_utils import PatchEmbed, Block
from defaults import MODEL_DEFAULTS

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        mask_ratio,
        img_size,
        in_chans,
        num_frames,
        patch_size,
        t_patch_size,
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        decoder_embed_dim,
        decoder_depth,
        decoder_num_heads,
        mlp_ratio,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,        
        patch_embed=PatchEmbed,
        no_qkv_bias=False,
        sep_pos_embed=True,
        trunc_init=False,
        cls_embed=True,
        **model_arguments
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.in_chans = in_chans
        self.patch_embed = patch_embed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            num_frames=num_frames,
            t_patch_size=t_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], encoder_embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], encoder_embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, encoder_embed_dim),    # would not differenciate between spatial and temporal when adding positional information
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    encoder_embed_dim,
                    encoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(encoder_depth)
            ]
        )
        self.norm = norm_layer(encoder_embed_dim)

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            t_patch_size * patch_size**2 * in_chans,
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        # I think the original 3 was hardcoded for the number of channels -> we only have 1 channel
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, C, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.patch_embed.t_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, self.in_chans, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * self.in_chans))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def unpatchify(self, x):
        # I think the original 3 was hardcoded for the number of channels -> we only have 1 channel
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, self.in_chans))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, self.in_chans, T, H, W))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim     #input is a batch of sequences of Length L, where each element in the sequence has a channel dimension D
        len_keep = int(L * (1 - mask_ratio))        

        noise = torch.rand(N, L)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        index = ids_keep.unsqueeze(-1).repeat(1, 1, D).to(x.device)
        x_masked = torch.gather(x, dim=1, index=index)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape # [Batch, num_patches_slices, (num_patches_height *num_patches_width), out_Channels]

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        x = x.view(N, -1, C) # scheint x_masked nicht zu verändern
        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            index = ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]).to(pos_embed.device)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=index,
            ) #also add the positional embeddings based on the c
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            index = ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]).to(pos_embed.device)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=index,
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size  # how many patches should be in the full pathched image

        # embed tokens
        x = self.decoder_embed(x)   # encoder_embed_dim to decoder_embed_dim
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)  #computes how many tokens are missing in the sequences
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token    adds the mask token to fill the sequence to the correct length
        x_ = x_.view([N, T * H * W, C])
        index = ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]).to(x_.device)
        x_ = torch.gather(
            x_, dim=1, index=index
        )  # unshuffle  # restores the original order of the sequences and adds the mask tokens at the correct position; maybe the correct ordering should be done beforehand??
        x = x_.view([N, T * H * W, C])
        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x) #scale every patch to the dimension it had in the original image

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, T, H, W]
        pred: [N, t*h*w, u*p*p*1]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """

        _imgs = torch.index_select(
            imgs,
            2,
            torch.linspace(
                0,
                imgs.shape[2] - 1,
                self.patch_embed.num_frames, # would use all the frames and not just a subset
            )
            .long()
            .to(imgs.device),
        )
        target = self.patchify(_imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per image-sequence in the batch
        mask = mask.view(loss.shape).to(loss.device)
        inv_mask = 1 - mask

        #loss = ((loss * mask).sum()+1) ** 2 / mask.sum() + (loss * inv_mask).sum() / mask.sum() # mean loss on removed patches
        #loss = loss.sum()
        loss = (loss*mask).sum() / mask.sum() # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*1]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    


class LitMaskedAutoencoder(L.LightningModule):

    def __init__(
        self,
        **model_arguments,
    ):
        super().__init__()
        self.autoencoder = MaskedAutoencoderViT(**model_arguments)
        self.learning_rate = model_arguments["learning_rate"]
        self.save_hyperparameters()

    @staticmethod  # register new arguments here
    def add_model_specific_args(parent_parser):
        """Adds model-specific arguments to the parser."""

        parser = parent_parser.add_argument_group("LitMaskedAutoencoder")
        parser.add_argument("--img_size", type=int, default=MODEL_DEFAULTS["MaskedAutoencoder"]["img_size"])
        parser.add_argument("--in_chans", type=int, default=MODEL_DEFAULTS["MaskedAutoencoder"]["in_chans"])
        parser.add_argument("--num_frames", type=int, default=MODEL_DEFAULTS["MaskedAutoencoder"]["num_frames"])
        parser.add_argument("--patch_size", type=int, default=MODEL_DEFAULTS["MaskedAutoencoder"]["patch_size"])
        parser.add_argument("--t_patch_size", type=int, default=MODEL_DEFAULTS["MaskedAutoencoder"]["t_patch_size"])
        parser.add_argument("--encoder_embed_dim", type=int, default=MODEL_DEFAULTS["MaskedAutoencoder"]["encoder_embed_dim"])
        parser.add_argument("--encoder_depth", type=int, default=MODEL_DEFAULTS["MaskedAutoencoder"]["encoder_depth"])
        parser.add_argument("--encoder_num_heads", type=int, default=MODEL_DEFAULTS["MaskedAutoencoder"]["encoder_num_heads"])
        parser.add_argument("--decoder_embed_dim", type=int, default=MODEL_DEFAULTS["MaskedAutoencoder"]["decoder_embed_dim"])
        parser.add_argument("--decoder_depth", type=int, default=MODEL_DEFAULTS["MaskedAutoencoder"]["decoder_depth"])
        parser.add_argument("--decoder_num_heads", type=int, default=MODEL_DEFAULTS["MaskedAutoencoder"]["decoder_num_heads"])
        parser.add_argument("--mlp_ratio", type=float, default=MODEL_DEFAULTS["MaskedAutoencoder"]["mlp_ratio"])
        parser.add_argument("--mask_ratio", type=float, default=MODEL_DEFAULTS["MaskedAutoencoder"]["mask_ratio"])        
        return parent_parser
    

    def configure_optimizers(self):
        
        #optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[15], gamma=0.1
        )
    
        return [optimizer], [lr_scheduler]

    def forward(self, x, mask_ratio=None):
        # suppose unifrom frame sampling and random resized cropping have already been applied to the images 
        return self.autoencoder(x, mask_ratio)
    
    
    def _calculate_loss(self, batch, mode="train"):
        imgs = batch
        
        loss, pred, mask = self.forward(imgs)
        self.log(f"{mode}_loss", loss, prog_bar=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="test")
    
class LitPretrainedVit(L.LightningModule):
    def __init__(
        self,
        pretrainedMae = None,
        **model_arguments,
    ):
        super().__init__()
        self.encoder = pretrainedMae.autoencoder.encoder
        self.learning_rate = model_arguments["learning_rate"]
        self.save_hyperparameters()
        self.classifier = nn.Linear(1024, 2)
     
    def forward(self, x):
        # scan has the orginal shape of nx1x128x128x128 with n being the batch size
        x = x.reshape(-1,1,16,128,128) #the batch now also contains the index dimension, because we split the 128 slice tensor into multiple 16 slice tensors
        latent, masks, ids_restore = self.encoder(x)
        latent = latent.reshape(-1,1,16,128,128)
        latent = latent[:, 0]
        latent = latent.reshape(-1, 8, 16)
        