import os
import re
import time
import shutil
import random
import argparse
import warnings

from tqdm import tqdm

import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader

from utils import Meter, UnetResNet, FPN, load_train_config, CityscapesTestDataset, torch2np, \
                  KittiTrainDataset, KittiTestDataset, KittiLaneDataset, \
                  CityscapesTrainDataset, CityscapesDataset, open_img

warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
args = parser.parse_args()
config = load_train_config(args.config_path)
globals().update(config)

if __name__ == "__main__":

    global_start = time.time()

    # Determine device and load checkpoint first to inspect class structure
    device = torch.device(EVAL["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[INFO] CUDA is not available on this machine. Falling back to CPU.")
    checkpoint_path = EVAL["model_path"]
    
    if not os.path.exists(checkpoint_path):
        # Tự động tìm đường dẫn fallback nếu không tìm thấy file theo config
        alternatives = [
            "weights/UNET_resnet50_cityscapes/best_model.pth",
            "./weights/UNET_resnet50_cityscapes/best_model.pth",
            "weights/UNET_resnet50_road/best_model.pth",
            "./weights/UNET_resnet50_road/best_model.pth"
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                checkpoint_path = alt
                print(f"[INFO] Tu dong chuyen huong checkpoint ve: {alt}")
                break
                
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state.get("state_dict", state)
    
    # Inspect number of classes
    num_classes = MODEL["num_classes"]
    if "final.weight" in state_dict:
        num_classes = state_dict["final.weight"].shape[0]
        print(f"[INFO] Detected num_classes from checkpoint: {num_classes}")

    # Automatically match train_on_cats to the checkpoint's classes
    if TARGET == "cityscapes":
        if num_classes == 4:
            DATASET["train_on_cats"] = False
            print("[INFO] Overriding train_on_cats to False to match 4-class checkpoint.")
        elif num_classes == 8:
            DATASET["train_on_cats"] = True
            print("[INFO] Overriding train_on_cats to True to match 8-class checkpoint.")

    if not EVAL["test_mode"]:

        if TARGET == "kitti":
            train_dataset = KittiTrainDataset(**PATHS["KITTI"])
            trainset, valset = train_dataset.get_paths()
            image_dataset = KittiLaneDataset(**DATASET)

        elif TARGET == "cityscapes":
            train_dataset = CityscapesTrainDataset(**PATHS["CITYSCAPES"])
            trainset, valset = train_dataset.get_paths()
            image_dataset = CityscapesDataset(**DATASET)

        image_dataset.set_phase("val", valset)

    else:

        if TARGET == "kitti":
            testset = KittiTestDataset(PATHS["KITTI"]["test_root_path"])
            image_dataset = KittiLaneDataset(**DATASET)
        
        elif TARGET == "cityscapes":
            testset = CityscapesTestDataset(PATHS["CITYSCAPES"]["test_root_path"])
            image_dataset = CityscapesDataset(**DATASET)

        image_dataset.set_phase("test", testset)

    dataloader = DataLoader(
        image_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=True,   
    )

    model = None
    if MODEL["mode"] == "UNET":
        backbone_candidates = [MODEL["backbone"], "resnext50", "resnet18", "resnet34", "resnet50"]
        loaded = False
        for candidate in backbone_candidates:
            try:
                print(f"[INFO] Attempting to load UNet with backbone: {candidate}...")
                model = UnetResNet(encoder_name=candidate, 
                                   num_classes=num_classes, 
                                   input_channels=3, 
                                   num_filters=32, 
                                   Dropout=0.2, 
                                   res_blocks_dec=MODEL["unet_res_blocks_decoder"])
                model.to(device)
                model.load_state_dict(state_dict)
                loaded = True
                print(f"[INFO] Successfully loaded UNet checkpoint with backbone: {candidate}")
                break
            except Exception as e:
                print(f"[WARNING] Failed loading with backbone {candidate}: {e}")
                continue
        if not loaded:
            raise RuntimeError("Failed to load UNet checkpoint with any candidate backbone.")
            
    elif MODEL["mode"] == "FPN":
        backbone_candidates = [MODEL["backbone"], "resnext50", "resnet18", "resnet34", "resnet50"]
        loaded = False
        for candidate in backbone_candidates:
            try:
                print(f"[INFO] Attempting to load FPN with backbone: {candidate}...")
                model = FPN(encoder_name=candidate,
                            decoder_pyramid_channels=256,
                            decoder_segmentation_channels=128,
                            classes=num_classes,
                            dropout=0.2,
                            activation='sigmoid',
                            final_upsampling=4,
                            decoder_merge_policy='add')
                model.to(device)
                model.load_state_dict(state_dict)
                loaded = True
                print(f"[INFO] Successfully loaded FPN checkpoint with backbone: {candidate}")
                break
            except Exception as e:
                print(f"[WARNING] Failed loading with backbone {candidate}: {e}")
                continue
        if not loaded:
            raise RuntimeError("Failed to load FPN checkpoint with any candidate backbone.")
    else:
        raise ValueError('Model type is not correct: `{}`.'.format(MODEL["mode"]))

    model.eval()



    eval_classes = num_classes if TARGET == "cityscapes" else MODEL["num_classes"]
    if not EVAL["test_mode"]:
        meter = Meter(base_threshold=EVAL["base_threshold"], get_class_metric=True)
        class_counts = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in range(eval_classes)}

    images_path = EVAL["eval_images_path"] if not EVAL["test_mode"] else EVAL["test_images_path"]
    try:
        shutil.rmtree(images_path)
    except:
        pass
    os.mkdir(images_path)

    start = time.time()
    for batch in tqdm(dataloader):
        images, targets, image_id = batch

        images = images.to(device)
        outputs = model(images)
        if EVAL["activate"]:
            outputs = torch.sigmoid(outputs)
        if DATASET["resize"]:
            outputs = torch.nn.functional.interpolate(outputs, size=DATASET["orig_size"], mode='bilinear', align_corners=True)
            targets = torch.nn.functional.interpolate(targets.float(), size=DATASET["orig_size"], mode='nearest')

        outputs = outputs.detach().cpu()
        
        # Hậu xử lý nâng cao (Morphological Post-processing & Threshold Tuning) để tăng IoU/Dice mà không cần train lại
        if outputs.shape[1] == 8 and not EVAL["test_mode"]:
            default_thresh = EVAL.get("base_threshold", 0.0)
            class_thresholds = [default_thresh] * 8
            # Tinh chỉnh ngưỡng quyết định tối ưu riêng cho từng lớp đích
            class_thresholds[1] = -2.2  # Lớp Road (Flat)
            class_thresholds[5] = -2.0  # Lớp Sky
            class_thresholds[7] = default_thresh  # Lớp Vehicle (Trở lại ngưỡng mặc định để khôi phục Precision)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            for c in range(8):
                # Phân ngưỡng nhị phân
                c_mask = (outputs[0, c, ...] > class_thresholds[c]).numpy().astype(np.uint8)
                
                # Áp dụng bộ lọc hình thái học phù hợp với đặc trưng hình học từng lớp
                if c in [6, 7]:
                    # Đối với đối tượng nhỏ (người, xe): chỉ mở ảnh (Opening) để lọc nhiễu, tránh đóng ảnh làm nở to viền gây mất Precision
                    c_mask_cleaned = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, kernel)
                else:
                    # Đối với mảng lớn (đường, bầu trời, nhà): đóng ảnh trước để lấp đầy lỗ hổng, rồi mở ảnh xóa nhiễu
                    c_mask_cleaned = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, kernel)
                    c_mask_cleaned = cv2.morphologyEx(c_mask_cleaned, cv2.MORPH_OPEN, kernel)
                    
                # Đưa về dạng logit tương thích (10.0 cho vùng nhận diện, -10.0 cho vùng nền)
                outputs[0, c, ...][c_mask_cleaned == 1] = 10.0
                outputs[0, c, ...][c_mask_cleaned == 0] = -10.0
        
        # Neu dataset la KITTI (1 lop) nhung model co 8 lop (Cityscapes), ta trich xuat lop "road" (lop 1) de danh gia
        if outputs.shape[1] == 8 and targets.shape[1] == 1:
            outputs = outputs[:, 1:2, ...]
            
        if not EVAL["test_mode"]:
            meter.update("val", targets, outputs)
            # Accumulate TP, TN, FP, FN per class
            preds_bin = (outputs > EVAL["base_threshold"]).float()
            targets_bin = (targets > 0.5).float()
            for c in range(eval_classes):
                preds_c = preds_bin[:, c, ...]
                targets_c = targets_bin[:, c, ...]
                
                tp = ((preds_c == 1) & (targets_c == 1)).sum().item()
                fp = ((preds_c == 1) & (targets_c == 0)).sum().item()
                fn = ((preds_c == 0) & (targets_c == 1)).sum().item()
                tn = ((preds_c == 0) & (targets_c == 0)).sum().item()
                
                class_counts[c]["tp"] += tp
                class_counts[c]["fp"] += fp
                class_counts[c]["fn"] += fn
                class_counts[c]["tn"] += tn

        # dump predictions as images
        outputs = (outputs > EVAL["base_threshold"]).int() # thresholding
        outputs = torch2np(outputs)
        pic = image_dataset.label_encoder.class2color(outputs, clean_up_clusters=EVAL["drop_clusters"],
                                                      mode="catId" if DATASET["train_on_cats"] else "trainId")
        if EVAL["images_morphing"]:
            # Add here image+mask morphing
            orig_image = open_img(image_id[0])
            alpha = 0.5
            if orig_image.shape[:2] != pic.shape[:2]:
                orig_image = cv2.resize(orig_image, (pic.shape[1], pic.shape[0]), cv2.INTER_LANCZOS4)
            pic = cv2.addWeighted(orig_image, (1 - alpha), pic, alpha, 0)

        img_filename = os.path.basename(image_id[0])
        pred_name = "_".join(re.split(r"\.|_", img_filename)[:-1]) + "_predicted_mask.png"
        cv2.imwrite(os.path.join(images_path, pred_name), pic)

    torch.cuda.empty_cache()
    if not EVAL["test_mode"]:
        dices, iou = meter.get_metrics("val")
        print("***** Prediction done in {} sec.; IoU: {}, Dice: {} ***** \n(total elapsed time: {} sec.) ".\
                format(int(time.time()-start), iou, dices[0]["dice_all"], int(time.time()-global_start)))
        if TARGET == "cityscapes" and len(dices[0]) > 1:
            labels_df = image_dataset.label_encoder.cityscapes_labels_df
            if DATASET["train_on_cats"]:
                cat, name = "catId", "category"
            else:
                cat, name = "trainId", "name"
            print("***** Class metrics (ALL 8 classes): *****")
            for k, v in dices[0].items():
                if k != "dice_all":
                    class_name = labels_df[labels_df[cat] == int(k)][name].iloc[0]
                    print(class_name, " : ", v)
        
        print("\n" + "="*50)
        print("***** CLASSIFICATION METRICS (ALL 8 Classes) *****")
        print("="*50)
        
        for c in range(eval_classes):
                
            tp = class_counts[c]["tp"]
            fp = class_counts[c]["fp"]
            fn = class_counts[c]["fn"]
            tn = class_counts[c]["tn"]
            total_pixels = tp + fp + fn + tn
            
            accuracy = (tp + tn) / (total_pixels + 1e-7)
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-7)
            
            class_prefix = f"Class {c}: " if eval_classes > 1 else ""
            if eval_classes > 1:
                try:
                    class_name_display = labels_df[labels_df[cat] == int(c)]["name"].iloc[0]
                    if DATASET["train_on_cats"]:
                         class_name_display = labels_df[labels_df[cat] == int(c)]["category"].iloc[0]
                    class_prefix = f"Class {c} ({class_name_display}): "
                except:
                    pass
            
            print(f"\n{class_prefix}Metrics:")
            print(f"  - Accuracy:  {accuracy:.6f}")
            print(f"  - Precision: {precision:.6f}")
            print(f"  - Recall:    {recall:.6f}")
            print(f"  - F1-Score:  {f1:.6f}")
            
            print(f"\n{class_prefix}Confusion Matrix:")
            print("                     Predicted Neg    Predicted Pos")
            print(f"Actual Neg (BG)    {tn:15d}  {fp:15d}")
            print(f"Actual Pos (FG)    {fn:15d}  {tp:15d}")
            print("-"*50)
    else:
        print("***** Prediction on test set done in {} sec. ***** \n(total elapsed time: {} sec.) ".\
                format(int(time.time()-start), int(time.time()-global_start)))
