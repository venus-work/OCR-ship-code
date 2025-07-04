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

# Constants
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
VIDEO_EXTENSIONS = ["mp4", "mov", "avi", "mkv"]

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
        Initialize PaddleOCR.
        Args:
            lang: Language for OCR (en, ch, vi, etc.)
            use_gpu: Whether to use GPU for OCR
        """
        self.ocr = PaddleOCR(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)

    def inference(self, img, confidence_threshold=0.6):
        """
        Perform OCR on the image.
        Args:
            img: Input image (numpy array)
            confidence_threshold: Minimum confidence for text detection
        Returns:
            List of detected text with bounding boxes and confidence scores
        """
        result = self.ocr.ocr(img)[0]
        ocr_results = [
            {'bbox': bbox, 'text': text, 'confidence': conf}
            for text, bbox, conf in zip(result["rec_texts"], result["rec_boxes"], result["rec_scores"]) if conf > confidence_threshold
        ]
        return ocr_results
    
    def visualize(self, img, ocr_results):
        """
        Draw OCR results on the image.
        Args:
            img: Input image
            ocr_results: OCR detection results
        Returns:
            Image with OCR results drawn
        """
        result_img = img.copy()
        for result in ocr_results:
            bbox = np.array(result['bbox'], dtype=np.int32)
            cv2.polylines(result_img, [bbox], True, (0, 255, 0), 2)
            label = f"{result['text']} ({result['confidence']:.2f})"
            cv2.putText(result_img, label, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return result_img

class NanoDetPredictor:
    def __init__(self, cfg, model_path, logger, device="cuda:0", ocr_lang="en", ocr_threshold=0.6):
        self.cfg = cfg
        self.device = device
        self.ocr_threshold = ocr_threshold

        # Initialize NanoDet model
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        
        # Initialize OCR
        self.ocr_predictor = PaddleOCRPredictor(lang=ocr_lang, use_gpu="cuda" in device)
        logger.log(f"OCR initialized with language: {ocr_lang}")

    def inference(self, img):
        """
        Run inference on the image.
        Args:
            img: Input image
        Returns:
            Inference results
        """
        img_info = {"id": 0, "file_name": os.path.basename(img) if isinstance(img, str) else None}
        img = cv2.imread(img) if isinstance(img, str) else img
        img_info["height"], img_info["width"] = img.shape[:2]
        
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def crop_boat_regions(self, img, dets, class_names, score_thres=0.5):
        """
        Crop boat regions from the image.
        Args:
            img: Original image
            dets: Detection results
            class_names: List of class names
            score_thres: Score threshold for detections
        Returns:
            List of cropped boat images with coordinates
        """
        boat_crops = []
        boat_class_idx = class_names.index('boat') if 'boat' in class_names else None
        
        if boat_class_idx is None:
            print("Warning: 'Boat' class not found in class_names")
            return boat_crops
        
        dets = dets[8]  # Assuming 'boat' class is at index 8
        for detection in dets:
            x1, y1, x2, y2, score = detection[:5]
            if score >= score_thres:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    cropped_img = img[y1:y2, x1:x2]
                    boat_crops.append({'crop': cropped_img, 'bbox': (x1, y1, x2, y2), 'score': float(score)})
        return boat_crops

    def process_boats_with_ocr(self, img, dets, class_names, score_thres=0.5):
        """
        Process detected boats with OCR.
        Args:
            img: Original image
            dets: Detection results
            class_names: List of class names
            score_thres: Score threshold for detections
        Returns:
            Dictionary containing OCR results for each boat
        """
        boat_ocr_results = []
        boat_crops = self.crop_boat_regions(img, dets, class_names, score_thres)
        
        for i, boat_data in enumerate(boat_crops):
            ocr_results = self.ocr_predictor.inference(boat_data['crop'], self.ocr_threshold)
            boat_ocr_results.append({
                'boat_id': i,
                'bbox': boat_data['bbox'],
                'detection_score': boat_data['score'],
                'ocr_results': ocr_results,
                'crop': boat_data['crop']
            })
        return boat_ocr_results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        """
        Visualize the results and OCR text on the image.
        Args:
            dets: Detection results
            meta: Metadata for the image
            class_names: List of class names
            score_thres: Score threshold for detections
        Returns:
            Image with results and OCR
        """
        result_img = self.model.head.show_result(meta["raw_img"][0].copy(), dets, class_names, score_thres=score_thres, show=False)
        boat_ocr_results = self.process_boats_with_ocr(meta["raw_img"][0].copy(), dets, class_names, score_thres)
        
        saving_txt_boat_list = []
        for index, boat_result in enumerate(boat_ocr_results):
            bbox = boat_result['bbox']
            ocr_results = boat_result['ocr_results']
            saving_txt_boat_list.append([str(i) for i in np.array(bbox, dtype=np.int32).reshape(-1)])
            
            for ocr_result in ocr_results:
                ocr_bbox = np.array(ocr_result['bbox'], np.int32).reshape(-1, 2)
                label = f"OCR: {ocr_result['text']} ({ocr_result['confidence']:.2f})"
                adjusted_bbox = [[pt[0] + bbox[0], pt[1] + bbox[1]] for pt in ocr_bbox]
                pts = np.array(adjusted_bbox, dtype=np.int32)
                
                cv2.rectangle(result_img, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), (0, 255, 0), 1)
                cv2.putText(result_img, label, (adjusted_bbox[0][0], adjusted_bbox[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                saving_txt_boat_list[index] += [j for j in [ocr_result['text']] + ["\t".join(np.array(i).reshape(-1).astype(str)) for i in ocr_bbox]]
        
        return result_img, saving_txt_boat_list

def get_image_list(path):
    """
    Get a list of image files in the given directory.
    Args:
        path: Directory path
    Returns:
        List of image file paths
    """
    return [
        os.path.join(maindir, filename)
        for maindir, subdir, file_name_list in os.walk(path)
        for filename in file_name_list
        if os.path.splitext(filename)[1] in IMAGE_EXTENSIONS
    ]

def main():
    args = parse_args()
    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = NanoDetPredictor(cfg, args.model, logger, device="cuda", ocr_lang=args.ocr_lang, ocr_threshold=args.ocr_threshold)
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()

    if args.demo == "image":
        files = get_image_list(args.path) if os.path.isdir(args.path) else [args.path]
        files.sort()
        for image_name in files:
            meta, res = predictor.inference(image_name)
            result_image, saving_txt_boat_list = predictor.visualize(res[0], meta, cfg.class_names, 0.5)
            if args.save_result:
                save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
                mkdir(local_rank, save_folder)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                cv2.imwrite(save_file_name, result_image)
                with open(f"{save_file_name}.txt", "w+") as fp:
                    for saving_boat in saving_txt_boat_list:
                        fp.write("\t".join(saving_boat) + "\n")
                print(f"Saved result to: {save_file_name}")

    elif args.demo in ["video", "webcam"]:
        cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
        width, height, fps = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)
        save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        mkdir(local_rank, save_folder)
        save_path = os.path.join(save_folder, os.path.basename(args.path) if args.demo == "video" else "camera.mp4")
        print(f"save_path is {save_path}")
        
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
        index = 0
        
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                meta, res = predictor.inference(frame.copy())
                result_frame, saving_txt_boat_list = predictor.visualize(res[0].copy(), meta, cfg.class_names, 0.5)
                if args.save_result:
                    vid_writer.write(result_frame)
                    with open(f'{save_path}_{index}.txt', "w+") as fp:
                        for saving_boat in saving_txt_boat_list:
                            fp.write("\t".join(saving_boat) + "\n")
                    print(f"Saved result to: {save_path}_{index}")
                index += 1
                if index == 100:
                    break

if __name__ == "__main__":
    main()
