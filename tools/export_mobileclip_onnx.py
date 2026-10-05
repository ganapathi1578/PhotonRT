"""Export Apple MobileCLIP-S1 using timm's fastvit_mci1 to fixed-shape ONNX."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

def build_model(checkpoint: Path) -> nn.Module:
    import timm
    from timm.utils.model import reparameterize_model
    model = timm.create_model(
        "fastvit_mci1", pretrained=True,
        pretrained_cfg_overlay={"file": str(checkpoint)},
        num_classes=512, in_chans=3, exportable=True)
    model.eval()
    model = reparameterize_model(model)
    model.eval()
    return model

class Wrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(); self.model=model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.model(x), p=2.0, dim=-1, eps=1e-12)

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",type=Path,required=True); ap.add_argument("--output",type=Path,default=Path("mobileclip-s1.onnx")); ap.add_argument("--opset",type=int,default=18); args=ap.parse_args()
    if not args.checkpoint.is_file(): raise FileNotFoundError(args.checkpoint)
    model=Wrapper(build_model(args.checkpoint)).eval(); dummy=torch.zeros(1,3,256,256,dtype=torch.float32)
    with torch.inference_mode(): ref=model(dummy).numpy()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,dummy,str(args.output),input_names=["image"],output_names=["embedding"],opset_version=args.opset,do_constant_folding=True,dynamic_axes=None,dynamo=False,external_data=False)
    print(f"Wrote {args.output}"); print("input shape : [1,3,256,256]"); print("output shape: [1,512]"); print(f"checksum    : {float(np.sum(ref)):.8e}")
    try:
        import onnx; onnx.checker.check_model(onnx.load(str(args.output),load_external_data=False)); print("onnx check  : passed")
    except ImportError: print("onnx check  : skipped (pip install onnx)")

if __name__ == "__main__": main()
