import torch
import torch.nn as nn
import pytorch_lightning as pl
from loss import focal_loss, ciou_loss,get_lr_scheduler,set_optimizer_lr,get_lr
from mbv4_timm import CenterNet
from utils_bbox import decode_bbox, centernet_correct_boxes_xyxy
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
from typing import Dict, List, Tuple, Optional, Any
import cv2
import glob 
from torch.optim.lr_scheduler import LambdaLR

def decode_offsets_to_boxes(pred_offsets, stride=4):
    """
    pred_offsets: shape [B, H, W, 4]
                  storing [left, top, right, bottom] per pixel
    stride      : how many pixels in the input space per 1 step in the feature map

    Returns:
        decoded_boxes: shape [B, H, W, 4], with absolute corners in [x_min, y_min, x_max, y_max]
    """
    B, H, W, _ = pred_offsets.shape

    # 1) Create a grid of center coords in the original input space
    device = pred_offsets.device
    yv, xv = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32)
    )
    xv = xv * stride  # shape (H, W)
    yv = yv * stride

    # 2) Flatten
    xv = xv.view(1, -1)  # shape (1, H*W)
    yv = yv.view(1, -1)

    # 3) Flatten the offsets
    offsets_flat = pred_offsets.view(B, -1, 4)  # (B, H*W, 4)
    # L, T, R, B
    left_offset   = offsets_flat[..., 0]
    top_offset    = offsets_flat[..., 1]
    right_offset  = offsets_flat[..., 2]
    bottom_offset = offsets_flat[..., 3]

    # 4) For each pixel i,
    x_min = xv - left_offset  * stride
    x_max = xv + right_offset * stride
    y_min = yv - top_offset   * stride
    y_max = yv + bottom_offset* stride

    # 5) Combine into [x_min, y_min, x_max, y_max]
    decoded = torch.stack([x_min, y_min, x_max, y_max], dim=-1)  # (B, H*W, 4)

    # 6) Reshape back to (B, H, W, 4)
    decoded = decoded.view(B, H, W, 4)
    return decoded

class LightningCenterNet(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, int] = (512, 512),
        batch_size : int = 16,
        stride: int = 4,
        lr: float = 5e-4,
        min_lr: float = 5e-6,
        weight_decay: float = 0,
        lr_decay_type: str = "yolox_cos",
        epochs: int = 100,
        coco_gt_path: Optional[str] = None,
        val_data_path: Optional[str] = None,
        classes: Optional[List[str]] = None,
        ciou_weight: float = 5.0,
        eval_interval: int = 5 
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = CenterNet(num_classes)
        
        # Parameters
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.stride = stride
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.lr_decay_type = lr_decay_type
        self.epochs = epochs
        self.ciou_weight = ciou_weight
        self.eval_interval = eval_interval
        
        # COCO evaluation
        self.coco_gt_path = coco_gt_path
        self.val_data_path = val_data_path
        self.classes = classes
        if self.coco_gt_path and os.path.exists(self.coco_gt_path):
            self.cocoGt = COCO(self.coco_gt_path)
        else:
            self.cocoGt = None
        
        # Best mAP tracking
        self.best_map = 0.0
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        batch_images, batch_hms, batch_regs, batch_reg_masks = batch
        
        # Forward pass
        hm, pred_reg = self(batch_images)
        
        # Classification loss (focal loss)
        c_loss = focal_loss(hm, batch_hms)
        
        # Regression loss (CIoU loss)
        pred_reg = pred_reg.permute(0, 2, 3, 1).contiguous()
        pred_boxes = decode_offsets_to_boxes(pred_reg, stride=self.stride)
        gt_boxes = decode_offsets_to_boxes(batch_regs, stride=self.stride) 
        loss_ciou = ciou_loss(pred_boxes, gt_boxes, batch_reg_masks)
        
        # Total loss
        loss = c_loss + loss_ciou * self.ciou_weight
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_c_loss', c_loss, prog_bar=False)
        self.log('train_ciou_loss', loss_ciou, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_images, batch_hms, batch_regs, batch_reg_masks = batch
        
        # Forward pass
        hm, pred_reg = self(batch_images)
        
        # Classification loss (focal loss)
        c_loss = focal_loss(hm, batch_hms)
        
        # Regression loss (CIoU loss)
        pred_reg = pred_reg.permute(0, 2, 3, 1).contiguous()
        pred_boxes = decode_offsets_to_boxes(pred_reg, stride=self.stride)
        gt_boxes = decode_offsets_to_boxes(batch_regs, stride=self.stride) 
        loss_ciou = ciou_loss(pred_boxes, gt_boxes, batch_reg_masks)
        
        # Total loss
        loss = c_loss + loss_ciou * self.ciou_weight
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_c_loss', c_loss, prog_bar=False, sync_dist=True)
        self.log('val_ciou_loss', loss_ciou, prog_bar=False, sync_dist=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        # Only run COCO evaluation on main process
        current_epoch = self.current_epoch
        # Check if we should run evaluation this epoch
        should_evaluate = (current_epoch % self.eval_interval == 0) or (current_epoch == self.trainer.max_epochs - 1)
        if self.trainer.is_global_zero and self.cocoGt and should_evaluate:
            # Save model temporarily for evaluation
            temp_path = "temp_model_for_eval.pth"
            torch.save(self.model.state_dict(), temp_path)
            
            # Run COCO evaluation
            mean_ap = self.evaluate_coco(temp_path)
            
            # Log mAP
            self.log('val_mAP', mean_ap, prog_bar=True)
            
            # Save best model
            if mean_ap > self.best_map:
                self.best_map = mean_ap
                self.log('best_mAP', self.best_map)
                # Ensure the checkpoint directory exists
                if hasattr(self.trainer, 'checkpoint_callback') and hasattr(self.trainer.checkpoint_callback, 'dirpath'):
                    checkpoint_dir = self.trainer.checkpoint_callback.dirpath
                    if checkpoint_dir is not None:
                        # Make sure the directory exists
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        
                        # Save the best model
                        best_model_path = os.path.join(checkpoint_dir, f"best_model_mAP_{mean_ap:.4f}.pth")
                        torch.save(self.model.state_dict(), best_model_path)
                        
                        # Update the best model path in the checkpoint callback
                        self.trainer.checkpoint_callback.best_model_path = best_model_path
                    else:
                        print("Warning: Checkpoint directory not available yet, skipping best model save")
                else:
                    print("Warning: Checkpoint callback not available, skipping best model save")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        elif self.trainer.is_global_zero and self.cocoGt:
            pass
            #print(f"Skipping COCO evaluation at epoch {current_epoch} (will evaluate every {self.eval_interval} epochs)")
    
    def evaluate_coco(self, model_path):
        """Run COCO evaluation on the model"""
        if not self.cocoGt or not self.classes:
            return 0.0
        
        # Print some info about ground truth annotations
        print(f"COCO GT info: {len(self.cocoGt.imgs)} images, {len(self.cocoGt.anns)} annotations")
        print(f"COCO categories: {self.cocoGt.cats}")
            
        # Prepare for evaluation
        folder = self.val_data_path
        print('folder:', folder)
        print('cocogt path:',self.coco_gt_path)
        val_images_folder = os.path.join(folder, "val_images")
        print('val images folder:',val_images_folder)
        # Get validation images
        val_images = []
        for ext in ["*.jpg", "*.png", "*.JPG"]:
            val_images.extend(glob.glob(os.path.join(val_images_folder, ext)))
        
        print(f"Found {len(val_images)} validation images")
        if len(val_images) == 0:
            print(f"No validation images found in {val_images_folder}")
            return 0.0
        
        # Run inference on validation images
        self.model.eval()
        results = []
        
        #for image_path in val_images:
        for i in self.cocoGt.dataset["images"]:
            try:
                image_id = i["id"]
                image_path = os.path.join(val_images_folder, i["file_name"])
                # Check if this image_id exists in the COCO ground truth
                if image_id not in self.cocoGt.imgs:
                    print(f"Warning: Image ID {image_id} not found in COCO annotations")
                
                # Read and preprocess image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Preprocess image
                image_shape = np.array(image.shape[:2])
                image_data = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_LINEAR)
                image_data = image_data.astype('float32') / 255.0
                image_data = (image_data - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                image_data = np.transpose(image_data, (2, 0, 1))[None]
                
                # Run inference
                with torch.no_grad():
                    input_tensor = torch.from_numpy(image_data).float().to(self.device)
                    hm, wh = self.model(input_tensor)
                    
                    # Decode predictions
                    try:
                        outputs = decode_bbox(hm, wh, stride=self.stride, confidence=0.05, cuda=True)
                        
                        # Check if outputs is empty
                        if not outputs or len(outputs[0]) == 0:
                            print(f"No detections for image {image_id}")
                            continue
                            
                        results_boxes = centernet_correct_boxes_xyxy(outputs, self.input_shape, image_shape, False).cpu()
                        
                        # Format results for COCO
                        for box in results_boxes:
                            if len(box) < 6:  # Ensure box has all required values
                                continue
                                
                            x1, y1, x2, y2, conf, cls_id = box
                            
                            # Ensure box coordinates are valid
                            if x2 <= x1 or y2 <= y1:
                                continue
                                
                            # Ensure class_id is valid
                            class_id = int(cls_id) + 1  # COCO categories start from 1
                            if class_id not in self.cocoGt.cats:
                                print(f"Warning: Class ID {class_id} not in COCO categories")
                                continue
                                
                            results.append({
                                'image_id': image_id,
                                'category_id': class_id,
                                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                'score': float(conf)
                            })
                    except Exception as e:
                        print(f"Error in detection for image {image_path}: {e}")
                        import traceback
                        traceback.print_exc()
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        
        # Save results to file
        with open('detection_results.json', 'w') as f:
            json.dump(results, f)
        
        print(f"Generated {len(results)} detections across {len(val_images)} images")
        
        # Check if we have any detections
        if len(results) == 0:
            print("No detections found in validation images. Cannot perform COCO evaluation.")
            return 0.0
        
        # Evaluate with COCO API
        try:
            # First validate that our results format is correct
            #for r in results[:5]:  # Print first few results for debugging
            #    print(f"Sample result: {r}")
                
            # Load results into COCO API
            cocoDt = self.cocoGt.loadRes('detection_results.json')
            
            # Make sure image IDs match between GT and detections
            gt_img_ids = set(self.cocoGt.getImgIds())
            dt_img_ids = set(cocoDt.getImgIds())
            common_img_ids = gt_img_ids.intersection(dt_img_ids)
            
            print(f"GT has {len(gt_img_ids)} images, DT has {len(dt_img_ids)} images")
            print(f"Common images: {len(common_img_ids)}")
            
            if len(common_img_ids) == 0:
                print("No common images between ground truth and detections!")
                return 0.0
                
            # Run evaluation
            cocoEval = COCOeval(self.cocoGt, cocoDt, 'bbox')
            
            # Optional: restrict evaluation to only images with detections
            # cocoEval.params.imgIds = list(common_img_ids)
            
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            
            # Check if stats is available and has values
            if hasattr(cocoEval, 'stats') and len(cocoEval.stats) > 0:
                mean_ap = cocoEval.stats[0]  # mAP at IoU thresholds from .50 to .95
                print(f"mAP: {mean_ap:.4f}")
                return mean_ap
            else:
                print("COCO evaluation completed but stats are not available")
                return 0.0
        except Exception as e:
            print(f"COCO evaluation error: {e}")
            # Print more detailed error information
            import traceback
            traceback.print_exc()
            return 0.0
        
    
    def configure_optimizers(self):

        Init_lr = self.lr #5e-4
        Min_lr  = Init_lr * 0.01
        momentum = 0.9
        nbs             = 64
        lr_limit_max    = 5e-4 
        lr_limit_min    = 2.5e-4 
        Init_lr_fit     = min(max(self.batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(self.batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=Init_lr_fit,
            betas = (momentum, 0.999),
            weight_decay=self.weight_decay
        )


        #total_iters = self.trainer.estimated_stepping_batches
        
        # Learning rate scheduler
        if self.lr_decay_type == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=self.min_lr
            )
        elif self.lr_decay_type == "yolox_cos":
            # Get the learning rate scheduler functions
            lr_scheduler_func = get_lr_scheduler(
                lr_decay_type='cos',
                lr=Init_lr_fit,
                min_lr=Min_lr_fit,
                total_iters=self.epochs  #total_iters
            )
            #scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_scheduler_func(epoch))
            self.lr_scheduler_func = lr_scheduler_func
            
            return optimizer
            
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.epochs // 10,
                gamma=0.1
            )
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def on_train_epoch_start(self):
        # Only apply manual LR scheduling if using yolox_cos
        if self.lr_decay_type == "yolox_cos":
            # Get current epoch
            current_epoch = self.current_epoch
            # Calculate the new learning rate based on the current epoch
            set_optimizer_lr(self.optimizers(), self.lr_scheduler_func, current_epoch)
            current_lr = get_lr(self.optimizers())
            # Log the learning rate
            self.log('learning_rate', current_lr, prog_bar=True)
