import argparse
import os
import time

import cv2
import torch
import numpy as np
from paddleocr import PaddleOCR

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("--config", help="model config file path")
    parser.add_argument("--model", help="model file path")
    parser.add_argument("--path", default="./demo", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--ocr_lang", 
        default="en", 
        help="OCR language (en, ch, vi, etc.)"
    )
    parser.add_argument(
        "--ocr_threshold", 
        type=float, 
        default=0.6, 
        help="OCR confidence threshold"
    )
    args = parser.parse_args()
    return args


class PaddleOCRPredictor:
    def __init__(self, lang='en', use_gpu=True):
        """
        Initialize PaddleOCR
        Args:
            lang: Language for OCR (en, ch, vi, etc.)
            use_gpu: Whether to use GPU for OCR
        """
        self.ocr = PaddleOCR(
            # use_angle_cls=True, 
            lang=lang, 
            use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
            use_doc_unwarping=False, # Disables text image rectification model via this parameter
            use_textline_orientation=False
            # use_gpu=use_gpu,
            # show_log=False
        )
    
    def inference(self, img, confidence_threshold=0.6):
        """
        Perform OCR on image
        Args:
            img: Input image (numpy array)
            confidence_threshold: Minimum confidence for text detection
        Returns:
            List of detected text with bounding boxes and confidence scores
        """
        result = self.ocr.ocr(img)[0]
        # print(result)
        ocr_results = []
        for text, bbox, conf in zip(result["rec_texts"], result["rec_boxes"], result["rec_scores"]):
            if conf > confidence_threshold:
                # pts = np.array(bbox, dtype=np.int32).reshape(-1, 2)
                # cv2.rectangle(img, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), (0, 255, 0), 1)
                # cv2.polylines(img, [np.array(bbox, dtype=np.int32).reshape(-1, 2)], True, (0, 255, 0), 2)

                # cv2.polylines(img, [np.array(bbox, dtype=np.int32).reshape(-1, 2)], (0, 255, 0), 1, isClosed=1)
                # cv2.imwrite("debug.jpg", img)
                ocr_results.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': conf
                })
        
        return ocr_results

    
    def visualize(self, img, ocr_results):
        """
        Draw OCR results on image
        Args:
            img: Input image
            ocr_results: OCR detection results
        Returns:
            Image with OCR results drawn
        """
        result_img = img.copy()
        
        for result in ocr_results:
            bbox = result['bbox']
            text = result['text']
            confidence = result['confidence']
            
            # Convert bbox to integer coordinates
            pts = np.array(bbox, dtype=np.int32)
            
            # Draw bounding box
            cv2.polylines(result_img, [pts], True, (0, 255, 0), 2)
            
            # Draw text and confidence
            label = f"{text} ({confidence:.2f})"
            cv2.putText(result_img, label, (int(bbox[0][0]), int(bbox[0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return result_img


class NanoDetPredictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0", ocr_lang="en", ocr_threshold=0.6):
        self.cfg = cfg
        self.device = device
        self.ocr_threshold = ocr_threshold
        
        # Initialize NanoDet model
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        
        # Initialize OCR
        self.ocr_predictor = PaddleOCRPredictor(lang=ocr_lang, use_gpu="cuda" in device)
        logger.log(f"OCR initialized with language: {ocr_lang}")

    def debug_detection_format(self, dets):
        """
        Debug method to understand detection format
        """
        print("=== DETECTION FORMAT DEBUG ===")
        # print(f"Type: {type(dets)}")
        
        if hasattr(dets, '__len__'):
            print(f"Length: {len(dets)}")
        
        if isinstance(dets, (list, tuple)):
            for i, det in enumerate(dets):
                print(f"dets[{i}] - Type: {type(det)}")
                if hasattr(det, 'shape'):
                    print(f"dets[{i}] - Shape: {det.shape}")
                elif hasattr(det, '__len__'):
                    print(f"dets[{i}] - Length: {len(det)}")
                    
                # Try to show first few elements if possible
                try:
                    if hasattr(det, '__getitem__'):
                        print(f"dets[{i}] - First element: {det[0] if len(det) > 0 else 'Empty'}")
                except:
                    pass
        else:
            if hasattr(dets, 'shape'):
                print(f"Shape: {dets.shape}")
            print(f"Content preview: {dets}")
        print("=== END DEBUG ===")

    def crop_boat_regions(self, img, dets, class_names, score_thres=0.5):
        """
        Crop boat regions from image for OCR processing
        Args:
            img: Original image
            dets: Detection results from NanoDet
            class_names: List of class names
            score_thres: Score threshold for detections
        Returns:
            List of cropped boat images with their coordinates
        """
        boat_crops = []
        
        # Find boat class index
        boat_class_idx = None
        for idx, class_name in enumerate(class_names):
            if class_name.lower() == 'boat':
                boat_class_idx = idx
                break
        
        if boat_class_idx is None:
            print("Warning: 'Boat' class not found in class_names")
            # Print available classes for debugging
            print(f"Available classes: {class_names}")
            return boat_crops
        
        # Debug: Print detection format
        # print(f"Detection format debug - type: {type(dets)}, length: {len(dets) if hasattr(dets, '__len__') else 'N/A'}")
        
        # Handle different detection formats
        try:
            # NanoDet typically returns detections as numpy arrays or tensors
            if hasattr(dets, 'cpu'):
                dets = dets.cpu().numpy()

            dets = dets[8] # 8 is class index of boat
            for detection in dets:
                x1, y1, x2, y2, score = detection[:5]
                if score >= score_thres:
                    # Crop boat region
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # Ensure coordinates are within image bounds
                    h, w = img.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        cropped_img = img[y1:y2, x1:x2]
                        boat_crops.append({
                            'crop': cropped_img,
                            'bbox': (x1, y1, x2, y2),
                            'score': float(score)
                        })
        except Exception as e:
            print(f"Error processing detections: {e}")
            print(f"Detection structure: {dets}")
            
        return boat_crops

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def process_boats_with_ocr(self, img, dets, class_names, score_thres=0.5):
        """
        Process detected boats with OCR
        Args:
            img: Original image
            dets: Detection results
            class_names: List of class names
            score_thres: Score threshold for detections
        Returns:
            Dictionary containing OCR results for each boat
        """
        boat_ocr_results = []
        
        # Debug detection format first
        
        # Get boat crops
        boat_crops = self.crop_boat_regions(img, dets, class_names, score_thres)
        
        print(f"Found {len(boat_crops)} boat(s) for OCR processing")
        
        for i, boat_data in enumerate(boat_crops):
            crop_img = boat_data['crop']
            bbox = boat_data['bbox']
            score = boat_data['score']
            
            # Perform OCR on cropped boat image
            ocr_results = self.ocr_predictor.inference(crop_img, self.ocr_threshold)
            
            if ocr_results:
                print(f"Boat {i+1} OCR results:")
                for ocr_result in ocr_results:
                    print(f"  Text: '{ocr_result['text']}' (confidence: {ocr_result['confidence']:.3f})")
            else:
                print(f"Boat {i+1}: No text detected")
            
            boat_ocr_results.append({
                'boat_id': i,
                'bbox': bbox,
                'detection_score': score,
                'ocr_results': ocr_results,
                'crop': crop_img
            })
        
        return boat_ocr_results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        
        # Original visualization
        result_img = self.model.head.show_result(
            meta["raw_img"][0].copy(), dets, class_names, score_thres=score_thres, show=False
        )
        # Process boats with OCR
        boat_ocr_results = self.process_boats_with_ocr(
            meta["raw_img"][0].copy(), dets, class_names, score_thres
        )
        
        saving_txt_boat_list = []
        # Draw OCR results on the main image
        for index, boat_result in enumerate(boat_ocr_results):
            bbox = boat_result['bbox']
            ocr_results = boat_result['ocr_results']
            saving_txt_boat_list.append([str(i) for i in np.array(bbox, dtype=np.int32).reshape(-1)])
            # Draw boat OCR results
            for ocr_result in ocr_results:
                # Convert OCR bbox coordinates to original image coordinates
                ocr_bbox = ocr_result['bbox']
                saving_txt_ocr_bbox = ocr_bbox.copy()
                ocr_bbox = np.array(ocr_bbox, np.int32).reshape(-1, 2)
                text = ocr_result['text']
                confidence = ocr_result['confidence']
                
                # Adjust OCR coordinates relative to boat bbox
                x1, y1, x2, y2 = bbox
                adjusted_bbox = []
                for point in ocr_bbox:
                    adj_x = int(point[0] + x1)
                    adj_y = int(point[1] + y1)
                    adjusted_bbox.append([adj_x, adj_y])
                
                # Draw OCR text box
                pts = np.array(adjusted_bbox, dtype=np.int32)
                cv2.rectangle(result_img, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), (0, 255, 0), 1)
                # cv2.polylines(result_img, [pts], True, (255, 255, 0), 2)
                
                # Draw OCR text
                label = f"OCR: {text} ({confidence:.2f})"
                cv2.putText(result_img, label, 
                           (adjusted_bbox[0][0], adjusted_bbox[0][1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                saving_txt_boat_list[index] += [j for j in [text]+[str(i) for i in saving_txt_ocr_bbox]]
                cv2.imwrite("debug2.jpg", result_img)

        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img, saving_txt_boat_list


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


def main():
    args = parse_args()
    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = NanoDetPredictor(
        cfg, args.model, logger, 
        device="cuda", 
        ocr_lang=args.ocr_lang,
        ocr_threshold=args.ocr_threshold
    )
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()
    
    if args.demo == "image":
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for image_name in files:
            meta, res = predictor.inference(image_name)
            result_image, saving_txt_boat_list = predictor.visualize(res[0], meta, cfg.class_names, 0.5)
            if args.save_result:
                save_folder = os.path.join(
                    cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                )
                mkdir(local_rank, save_folder)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                cv2.imwrite(save_file_name, result_image)
                with open(save_file_name+".txt", "w+") as fp1:
                    for saving_boat in saving_txt_boat_list:
                        saving_boat = "\t".join(saving_boat)
                        fp1.write(saving_boat+"\n")
                print(f"Saved result to: {save_file_name}")
            ch = 27
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
                
    elif args.demo == "video" or args.demo == "webcam":
        cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        save_folder = os.path.join(
            cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        mkdir(local_rank, save_folder)
        save_path = (
            os.path.join(save_folder, args.path.replace("\\", "/").split("/")[-1])
            if args.demo == "video"
            else os.path.join(save_folder, "camera.mp4")
        )
        print(f"save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        index = 0
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                meta, res = predictor.inference(frame.copy())
                result_frame, saving_txt_boat_list = predictor.visualize(res[0].copy(), meta, cfg.class_names, 0.5)
                if args.save_result:
                    # save_folder = os.path.join(
                    #     cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                    # )
                    # mkdir(local_rank, save_folder)
                    # save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                    vid_writer.write(result_frame)
                    # cv2.imwrite(save_file_name, result_image)
                    with open(f'{save_path}_{index}.txt', "w+") as fp1:
                        for saving_boat in saving_txt_boat_list:
                            saving_boat = "\t".join(saving_boat)
                            fp1.write(saving_boat+"\n")
                    print(f"Saved result to: {save_path}")
                index += 1
                if index == 100:
                    break



                # meta, res = predictor.inference(frame)
                # result_frame = predictor.visualize(res[0], meta, cfg.class_names, 0.5)
                # if args.save_result:
                #     vid_writer.write(result_frame)
                # ch = 27 
                # if ch == 27 or ch == ord("q") or ch == ord("Q"):
                #     break
            else:
                break


if __name__ == "__main__":
    main()