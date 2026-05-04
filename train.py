import shutil
from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':
    RESUME = True   # set False to start a fresh run

    model = YOLO('runs/detect/runs/biminspect-det-v10-balanced2/weights/last.pt' if RESUME else 'yolov8m.pt')
    results = model.train(
        data    = 'models/configs/dataset_multiclass.yaml',
        epochs  = 150,
        imgsz   = 640,
        batch   = 12,           # YOLOv8m at 640px: fits RTX 3070 8 GB
        cache   = False,
        amp     = True,
        workers = 0,
        device  = 0,
        project = 'runs',
        name    = 'biminspect-det-v10-balanced2',
        exist_ok = RESUME,
        resume   = RESUME,
        patience = 30,
        seed     = 42,
        lr0      = 0.01,
        lrf      = 0.01,
        warmup_epochs = 3,
        hsv_h   = 0.015,
        hsv_s   = 0.7,
        hsv_v   = 0.4,
        flipud  = 0.3,
        fliplr  = 0.5,
        mosaic  = 0.5,          # reduced — tiles already provide spatial variety
        mixup   = 0.1,
        close_mosaic = 10,

        # ── Anti-overfitting
        weight_decay    = 0.001,
        dropout         = 0.1,
        label_smoothing = 0.05,
    )

    best_src  = Path('runs/detect/runs/biminspect-det-v10-balanced2/weights/best.pt')
    best_dest = Path('models/weights/best_detection.pt')
    if best_src.exists():
        shutil.copy2(best_src, best_dest)
        print('Best weights saved:', best_dest)

    m    = results.results_dict
    prev = 0.5779  # v9 baseline (YOLOv8s, tiled)
    new  = m.get('metrics/mAP50(B)', 0)
    print(f'mAP50     : {new:.4f}  (prev: {prev:.4f}  delta: {new - prev:+.4f})')
    print(f'Precision : {m.get("metrics/precision(B)", "n/a")}')
    print(f'Recall    : {m.get("metrics/recall(B)", "n/a")}')
