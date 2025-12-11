from export_importable_onnx_only import export_model

export_model(
    weights=R"\\dl-trainer-4\D\Local\Yolo\Yv7\yolov7\runs\train\Therm20251104_pts04_try4\weights\last.pt",
    img_size=640,
    batch_size=1,
    device="0",
    grid=True,        # or False, depending on what you want
    simplify=True,   # optional
    include_nms=False,
)