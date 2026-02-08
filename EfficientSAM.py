
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#论文：ReGLA: Efficient Receptive-Field Modeling with Gated Linear AttentionNetwork
#论文地址：https://arxiv.org/pdf/2602.05262
# ==================== 修复版 EfficientSAM ====================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SAMIEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x, context):
        B, N_q, C = x.shape
        N_kv = context.shape[1]
        q = self.q(x).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        return self.proj(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4, dim)
    def forward(self, x, context):
        x = x + self.cross_attn(self.norm1(x), context)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=8, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([CrossAttentionBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, masked_tokens, unmasked_tokens):
        x = masked_tokens
        for blk in self.blocks:
            x = blk(x, unmasked_tokens)
        return self.norm(x)


class SAMI(nn.Module):
    """SAM-Leveraged Masked Image Pretraining"""
    def __init__(self, img_size=224, patch_size=16, encoder_embed_dim=384, encoder_depth=12, encoder_num_heads=6, decoder_depth=8, decoder_num_heads=8, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = SAMIEncoder(img_size, patch_size, embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads)
        self.decoder = CrossAttentionDecoder(encoder_embed_dim, decoder_depth, decoder_num_heads)
        self.proj_head = nn.Linear(encoder_embed_dim, 1280)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=.02)
    
    def random_masking(self, x, mask_ratio):
        B, N, C = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_unmasked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_unmasked, mask, ids_restore, ids_keep
    
    def forward(self, x, sam_features=None):
        B = x.shape[0]
        x = self.encoder.patch_embed(x)
        N = x.shape[1]
        x = x + self.encoder.pos_embed
        x_unmasked, mask, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)
        x_encoded = x_unmasked
        for blk in self.encoder.blocks:
            x_encoded = blk(x_encoded)
        x_encoded = self.encoder.norm(x_encoded)
        mask_tokens = self.mask_token.expand(B, N - ids_keep.shape[1], -1)
        x_full = torch.cat([x_encoded, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2]))
        mask_bool = mask.bool()
        x_masked_list = [x_full[b, mask_bool[b]] for b in range(B)]
        x_masked = torch.stack(x_masked_list)
        x_reconstructed = self.decoder(x_masked, x_encoded)
        pred = self.proj_head(x_reconstructed)
        if sam_features is not None:
            target_list = [sam_features[b, mask_bool[b]] for b in range(B)]
            target = torch.stack(target_list)
            loss = F.mse_loss(pred, target)
            return loss, pred, mask
        return pred, mask


class SimpleMaskDecoder(nn.Module):
    def __init__(self, transformer_dim=384, num_multimask_outputs=3):
        super().__init__()
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(num_multimask_outputs, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, 2, 2),
            nn.GroupNorm(1, transformer_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, 2, 2),
            nn.GELU()
        )
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, transformer_dim // 4) for _ in range(num_multimask_outputs)])
        self.iou_prediction_head = MLP(transformer_dim, 256, num_multimask_outputs)
    def forward(self, image_embeddings):
        B, C, H, W = image_embeddings.shape
        iou_token_out = self.iou_token.weight.expand(B, -1, -1)
        mask_tokens_out = self.mask_tokens.weight.expand(B, -1, -1)
        upscaled_embedding = self.output_upscaling(image_embeddings)
        masks = []
        for i in range(self.num_multimask_outputs):
            hyper_in_i = self.output_hypernetworks_mlps[i](mask_tokens_out[:, i:i+1, :])
            b, c, h, w = upscaled_embedding.shape
            masks.append((hyper_in_i @ upscaled_embedding.view(b, c, h * w)).view(b, 1, h, w))
        masks = torch.cat(masks, dim=1)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


class EfficientSAM(nn.Module):
    def __init__(self, img_size=1024, patch_size=16, encoder_embed_dim=384, encoder_depth=12, encoder_num_heads=6):
        super().__init__()
        self.image_encoder = SAMIEncoder(img_size, patch_size, embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads)
        self.mask_decoder = SimpleMaskDecoder(encoder_embed_dim)
    def forward(self, images):
        image_embeddings = self.image_encoder(images)
        B, N, C = image_embeddings.shape
        H = W = int(N ** 0.5)
        image_embeddings = image_embeddings.transpose(1, 2).view(B, C, H, W)
        masks, iou_pred = self.mask_decoder(image_embeddings)
        return masks, iou_pred


# ==================== 测试 ====================

print("=" * 60)
print("EfficientSAM 测试套件")
print("=" * 60)

# 测试1: SAMI
print("\n1. SAMI预训练")
sami = SAMI(img_size=224, patch_size=16, encoder_embed_dim=192, encoder_depth=6, encoder_num_heads=3, decoder_depth=4, decoder_num_heads=4, mask_ratio=0.75)
B = 2
x = torch.randn(B, 3, 224, 224)
N = (224 // 16) ** 2
sam_features = torch.randn(B, N, 1280)
loss, pred, mask = sami(x, sam_features)
print(f"输入: {x.shape}")
print(f"SAM特征: {sam_features.shape}")
print(f"重建: {pred.shape}")
print(f"Mask: {mask.sum() / mask.numel():.2%}")
print(f"损失: {loss.item():.6f}")
print(f"参数量: {sum(p.numel() for p in sami.parameters()):,}")
loss.backward()
print("梯度: OK ✓")

# 测试2: EfficientSAM
print("\n2. EfficientSAM")
model = EfficientSAM(img_size=512, patch_size=16, encoder_embed_dim=192, encoder_depth=6, encoder_num_heads=3)  # 减小尺寸
x = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    masks, iou_pred = model(x)
print(f"输入: {x.shape}")
print(f"Masks: {masks.shape}")
print(f"IoU: {iou_pred.shape}")
params = sum(p.numel() for p in model.parameters())
sam_vit_h = 641_000_000
print(f"参数量: {params:,}")
print(f"相比SAM减少: {(1 - params/sam_vit_h)*100:.1f}% ✓")

# 测试3: 交叉注意力
print("\n3. 交叉注意力解码器")
decoder = CrossAttentionDecoder(embed_dim=192, depth=4, num_heads=4)
masked = torch.randn(2, 147, 192)
unmasked = torch.randn(2, 49, 192)
out = decoder(masked, unmasked)
print(f"输入: {masked.shape}, {unmasked.shape}")
print(f"输出: {out.shape} ✓")

# 测试4: 对比
print("\n4. 效率对比")
models = {'SAM ViT-H': 641, 'EfficientSAM-S': 38, 'EfficientSAM-Ti': 9.8, 'MobileSAM': 9.8}
print(f"{'Model':<20} {'Params (M)':<15} {'Reduction':<15}")
print("-" * 50)
baseline = models['SAM ViT-H']
for name, params in models.items():
    reduction = (1 - params / baseline) * 100
    print(f"{name:<20} {params:<15.1f} {reduction:<15.1f}%")
print("✓")

print("\n" + "=" * 60)
print("所有测试完成！")
print("=" * 60)
