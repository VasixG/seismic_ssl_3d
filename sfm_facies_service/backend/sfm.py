import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import timm
from tqdm import tqdm


def build_sfm_vit(model_size: str = "base", img_size: int = 224, in_chans: int = 1):
    if model_size == "base":
        embed_dim, depth, num_heads = 768, 12, 12
    elif model_size == "large":
        embed_dim, depth, num_heads = 1024, 24, 16
    else:
        raise ValueError("model_size must be 'base' or 'large'")

    vit = timm.models.vision_transformer.VisionTransformer(
        img_size=img_size,
        patch_size=16,
        in_chans=in_chans,
        num_classes=0,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return vit


def load_sfm_checkpoint(model: nn.Module, ckpt_path: str, map_location="cpu"):
    # PyTorch 2.6 defaults to weights_only=True; some checkpoints store argparse.Namespace.
    # We explicitly allow full loading here to support common training checkpoints.
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def extract_patch_tokens(vit: nn.Module, x: torch.Tensor):
    x_tok = vit.patch_embed(x)
    cls = vit.cls_token.expand(x_tok.shape[0], -1, -1)
    x_tok = torch.cat((cls, x_tok), dim=1)
    x_tok = x_tok + vit.pos_embed
    x_tok = vit.pos_drop(x_tok)

    for blk in vit.blocks:
        x_tok = blk(x_tok)

    x_tok = vit.norm(x_tok)
    patch_tokens = x_tok[:, 1:, :].contiguous()
    hp = x.shape[2] // vit.patch_embed.patch_size[0]
    wp = x.shape[3] // vit.patch_embed.patch_size[1]
    return patch_tokens, (hp, wp)


def robust_norm(img: np.ndarray, eps=1e-6):
    x = img.astype(np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    x = (x - med) / (1.4826 * mad + eps)
    x = np.clip(x, -6.0, 6.0)
    return x


def pad_to_multiple(img2d: np.ndarray, mult: int, pad_mode="edge"):
    h, w = img2d.shape
    h2 = int(math.ceil(h / mult) * mult)
    w2 = int(math.ceil(w / mult) * mult)
    pad_h = h2 - h
    pad_w = w2 - w
    if pad_h == 0 and pad_w == 0:
        return img2d, (0, 0)
    img_pad = np.pad(img2d, ((0, pad_h), (0, pad_w)), mode=pad_mode)
    return img_pad, (pad_h, pad_w)


@torch.no_grad()
def features_for_2d(
    vit: nn.Module,
    img2d: np.ndarray,
    tile_size: int,
    device: str,
    batch_tiles: int = 8,
):
    img = robust_norm(img2d)
    img, (ph, pw) = pad_to_multiple(img, tile_size, pad_mode="edge")
    h, w = img.shape

    pps = tile_size // 16
    h_pat = (h // tile_size) * pps
    w_pat = (w // tile_size) * pps
    d = vit.embed_dim
    feat_grid = np.zeros((h_pat, w_pat, d), dtype=np.float32)

    tiles, coords = [], []
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            tiles.append(img[y0:y0 + tile_size, x0:x0 + tile_size])
            coords.append((y0, x0))

    vit = vit.to(device)
    vit.eval()

    def flush(batch_np, batch_xy):
        x = np.stack(batch_np, axis=0)
        x = torch.from_numpy(x).unsqueeze(1).to(device)
        tok, (hp, wp) = extract_patch_tokens(vit, x)
        tok = tok.detach().cpu().numpy().reshape(len(batch_np), hp, wp, d)

        for i, (y0, x0) in enumerate(batch_xy):
            gy = (y0 // tile_size) * pps
            gx = (x0 // tile_size) * pps
            feat_grid[gy:gy + pps, gx:gx + pps, :] = tok[i]

    for i in tqdm(range(0, len(tiles), batch_tiles), desc="SFM tiles"):
        flush(tiles[i:i + batch_tiles], coords[i:i + batch_tiles])

    pad_pat_h = ph // 16
    pad_pat_w = pw // 16
    if pad_pat_h > 0:
        feat_grid = feat_grid[:-pad_pat_h, :, :]
    if pad_pat_w > 0:
        feat_grid = feat_grid[:, :-pad_pat_w, :]

    return feat_grid
