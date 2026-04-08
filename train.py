import shutil
from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    results = model.train(
        data    = 'models/configs/dataset_manual_only.yaml',
        epochs  = 150,
        imgsz   = 640,
        batch   = 32,
        cache   = True,
        amp     = True,
        workers = 0,
        device   = 0,
        project  = 'models/weights',
        name     = 'biminspect-det-v5-manual',
        exist_ok = False,
        resume   = False,
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
        mosaic  = 0.8,
        mixup   = 0.1,
        close_mosaic = 10,

        # ── Anti-overfitting
        weight_decay    = 0.001,   # penalises large weights
        dropout         = 0.1,     # regularises classification head
        label_smoothing = 0.05,    # reduces overconfidence on noisy labels
    )

    best_src  = Path('models/weights/biminspect-det-v5-manual/weights/best.pt')
    best_dest = Path('models/weights/best_detection.pt')
    if best_src.exists():
        shutil.copy2(best_src, best_dest)
        print('Best weights saved:', best_dest)

    m    = results.results_dict
    prev = 0.6170
    new  = m.get('metrics/mAP50(B)', 0)
    print(f'mAP50     : {new:.4f}  (prev: {prev:.4f}  delta: {new - prev:+.4f})')
    print(f'Precision : {m.get("metrics/precision(B)", "n/a")}')
    print(f'Recall    : {m.get("metrics/recall(B)", "n/a")}')
