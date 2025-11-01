# tools/gradcam.py
import torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _last_conv_layer(model: torch.nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last

def gradcam_on_image(model, image_tensor, target_index=None):
    """Returns (cam, target_index). image_tensor: [1,3,H,W]."""
    model.eval()
    feats = None; grads = None
    conv = _last_conv_layer(model)
    h1 = conv.register_forward_hook(lambda m, i, o: locals().update(feats:=o))
    def b_hook(m, gi, go):
        nonlocal grads; grads = go[0]
    h2 = conv.register_full_backward_hook(b_hook)

    out = model(image_tensor)
    if target_index is None:
        target_index = int(out.sigmoid().detach().cpu().numpy().argmax())
    loss = out[0, target_index]
    model.zero_grad(); loss.backward()

    alpha = grads.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
    cam = (alpha * feats).sum(dim=1).relu()[0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    h1.remove(); h2.remove()
    return cam, target_index

def overlay(rgb_np: np.ndarray, cam: np.ndarray, out_path: str, alpha: float = 0.35):
    plt.figure()
    plt.imshow(rgb_np.astype(np.uint8))
    plt.imshow((cam*255).astype(np.uint8), alpha=alpha)
    plt.axis("off"); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
