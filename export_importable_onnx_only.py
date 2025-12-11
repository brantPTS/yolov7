import argparse
import sys
import time
import warnings
from types import SimpleNamespace

sys.path.append("./")  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load, End2End
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from utils.add_nms import RegisterNMS


def run_export(opt: SimpleNamespace) -> None:
    """
    Core ONNX export logic.

    Expected fields on `opt`:
      weights, img_size, batch_size,
      dynamic, dynamic_batch, grid, end2end,
      max_wh, topk_all, iou_thres, conf_thres,
      device, simplify, include_nms
    """

    # Normalize img_size behavior (same as original script)
    if isinstance(opt.img_size, int):
        opt.img_size = [opt.img_size]
    else:
        opt.img_size = list(opt.img_size)

    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand [640] -> [640, 640]
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic

    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,640,640)

    # Update model (export-friendly activations)
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()

    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run
    if opt.include_nms:
        model.model[-1].include_nms = True
        y = None

    # ----------------------
    # ONNX export only
    # ----------------------
    try:
        import onnx

        print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
        f = opt.weights.replace(".pt", ".onnx")  # filename
        model.eval()
        output_names = ["classes", "boxes"] if y is None else ["output"]
        dynamic_axes = None

        # Dynamic shapes (width/height)
        if opt.dynamic:
            dynamic_axes = {
                "images": {0: "batch", 2: "height", 3: "width"},  # size(1,3,640,640)
                "output": {0: "batch", 2: "y", 3: "x"},
            }

        # Dynamic batch
        if opt.dynamic_batch:
            batch_symbol = "batch"
            dynamic_axes = {
                "images": {
                    0: batch_symbol,
                },
            }
            if opt.end2end and opt.max_wh is None:
                output_axes = {
                    "num_dets": {0: batch_symbol},
                    "det_boxes": {0: batch_symbol},
                    "det_scores": {0: batch_symbol},
                    "det_classes": {0: batch_symbol},
                }
            else:
                output_axes = {
                    "output": {0: batch_symbol},
                }
            dynamic_axes.update(output_axes)

        # For TensorRT / ORT end2end wrapper
        if opt.grid:
            if opt.end2end:
                print(
                    "\nStarting export end2end onnx model for %s..."
                    % ("TensorRT" if opt.max_wh is None else "onnxruntime")
                )
                model_ort = End2End(
                    model,
                    opt.topk_all,
                    opt.iou_thres,
                    opt.conf_thres,
                    opt.max_wh,
                    device,
                    len(labels),
                )
                model_to_export = model_ort
                if opt.end2end and opt.max_wh is None:
                    output_names = [
                        "num_dets",
                        "det_boxes",
                        "det_scores",
                        "det_classes",
                    ]
                    shapes = [
                        opt.batch_size,
                        1,
                        opt.batch_size,
                        opt.topk_all,
                        4,
                        opt.batch_size,
                        opt.topk_all,
                        opt.batch_size,
                        opt.topk_all,
                    ]
                else:
                    output_names = ["output"]
            else:
                # concat all detection heads
                model.model[-1].concat = True
                model_to_export = model
        else:
            model_to_export = model

        # Export ONNX
        torch.onnx.export(
            model_to_export,
            img,
            f,
            verbose=False,
            opset_version=12,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        # Fix dynamic shapes for end2end TensorRT variant
        if opt.end2end and opt.max_wh is None and opt.grid:
            for i in onnx_model.graph.output:
                for j in i.type.tensor_type.shape.dim:
                    j.dim_param = str(shapes.pop(0))

        # Optional ONNX simplification
        if opt.simplify:
            try:
                import onnxsim

                print("\nStarting to simplify ONNX...")
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, "assert check failed"
            except Exception as e:
                print(f"Simplifier failure: {e}")

        onnx.save(onnx_model, f)
        print("ONNX export success, saved as %s" % f)

        # Optional NMS plugin registration
        if opt.include_nms:
            print("Registering NMS plugin for ONNX...")
            mo = RegisterNMS(f)
            mo.register_nms()
            mo.save(f)

    except Exception as e:
        print("ONNX export failure: %s" % e)

    # Finish
    print(
        "\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron."
        % (time.time() - t)
    )


# -----------------------------
# High-level importable API
# -----------------------------
def export_model(
    weights: str = "./yolor-csp-c.pt",
    img_size=(640, 640),
    batch_size: int = 1,
    dynamic: bool = False,
    dynamic_batch: bool = False,
    grid: bool = False,
    end2end: bool = False,
    max_wh=None,
    topk_all: int = 100,
    iou_thres: float = 0.45,
    conf_thres: float = 0.25,
    device: str = "cpu",
    simplify: bool = False,
    include_nms: bool = False,
) -> None:
    """
    High-level API for ONNX-only export, for use from notebooks / other Python code.

    Example:
        from export_importable_onnx_only import export_model

        export_model(
            weights="../runs/train/exp/weights/best.pt",
            img_size=640,
            batch_size=1,
            device="0",
            simplify=True,
            grid=True,
        )
    """
    # Normalize img_size to match original expectations
    if isinstance(img_size, int):
        img_size = [img_size]
    else:
        img_size = list(img_size)

    opt = SimpleNamespace(
        weights=weights,
        img_size=img_size,
        batch_size=batch_size,
        dynamic=dynamic,
        dynamic_batch=dynamic_batch,
        grid=grid,
        end2end=end2end,
        max_wh=max_wh,
        topk_all=topk_all,
        iou_thres=iou_thres,
        conf_thres=conf_thres,
        device=device,
        simplify=simplify,
        include_nms=include_nms,
    )
    return run_export(opt)


# -----------------------------
# CLI glue
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (for command-line use)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="./yolor-csp-c.pt", help="weights path")
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="image size",  # height, width
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--dynamic", action="store_true", help="dynamic ONNX axes (H/W)")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="dynamic batch onnx for tensorrt and onnx-runtime",
    )
    parser.add_argument("--grid", action="store_true", help="export Detect() layer grid")
    parser.add_argument("--end2end", action="store_true", help="export end2end onnx")
    parser.add_argument(
        "--max-wh",
        type=int,
        default=None,
        help="None for tensorrt nms, int value for onnx-runtime nms",
    )
    parser.add_argument(
        "--topk-all",
        type=int,
        default=100,
        help="topk objects for every image",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="iou threshold for NMS (end2end)",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="conf threshold for NMS (end2end)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu",
    )
    parser.add_argument("--simplify", action="store_true", help="simplify onnx model")
    parser.add_argument(
        "--include-nms",
        action="store_true",
        help="attach NMS plugin to exported ONNX",
    )
    return parser


def main(argv=None) -> None:
    """CLI entrypoint."""
    parser = build_argparser()
    opt = parser.parse_args(argv)
    run_export(opt)


if __name__ == "__main__":
    main()
