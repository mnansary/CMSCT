import torch
from yolox.exp import get_exp
from torch2trt import TRTModule
import concurrent.futures
import cv2 
import numpy as np
from yolox.utils import postprocess
from loguru import logger
import time
from torch.cuda.amp import autocast 


def preproc_single(img, input_size, mean, std, swap=(2, 0, 1)):
    padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
 
    padded_img = padded_img[:, :, ::-1]  # Convert BGR to RGB
 
    # Normalize the image (scale and apply mean/std)
    padded_img /= 255.0
    padded_img = (padded_img - mean) / std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    
    return padded_img
 
def preproc_batch_concurrent(images, input_size, mean, std, swap=(2, 0, 1)):
    batch_size = len(images)
    
    # Create a list to store the futures
    processed_images = [None] * batch_size  # Initialize a list to hold the processed images
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each image for processing, passing the index to maintain order
        futures = {
            executor.submit(preproc_single, images[i], input_size, mean, std, swap): i
            for i in range(batch_size)
        }
 
        # Wait for all futures to complete and collect results in the original order
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]  # Retrieve the original index
            processed_images[index] = torch.tensor(future.result(), dtype=torch.float32)
 
    # Stack the processed images into a single tensor
    processed_images_batch = torch.stack(processed_images)
    
    return processed_images_batch


class Detector(object):
    def __init__(self,exp_file,trt_file,device=torch.device("cuda:0")):
        # fix exp params
        self.exp=get_exp(exp_file,exp_name="cmsct")
        self.device = device

        # load model
        self.model = self.exp.get_model().to(self.device)
        self.model.eval()
        self.model.head.decode_in_inference=False
        self.decoder=self.model.head.decode_outputs
        # trt model
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(trt_file))
        x = torch.ones((1, 3, self.exp.test_size[0], self.exp.test_size[1]), device=self.device)
        self.model(x)
        self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        logger.info("Detector Loaded")

    def inference_batch(self, images,batch_size=32):
        start=time.perf_counter()
        # preproc
        images=preproc_batch_concurrent(images,self.exp.test_size,self.rgb_means,self.std)
        images=images.float().to(self.device)
        
        all_outputs=[]
        with torch.no_grad():
            for i in range(0,len(images),batch_size):
                batch=images[i:i+batch_size]
                with autocast():
                    outputs = self.model(batch)
                    if self.decoder is not None:
                        outputs = self.decoder(outputs, dtype=outputs.type())
                all_outputs.append(outputs)
                torch.cuda.empty_cache()
        all_outputs=torch.cat(all_outputs,dim=0)
        all_outputs = postprocess(all_outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre)
        taken_time=time.perf_counter()-start
        logger.info(f"Processed images: {len(images)} FPS: {len(images)/taken_time}")
        return all_outputs


if __name__=="__main__":
    #from cmsct.detector import Detector
    from cmsct.constants import PERSONDET_TRT_MODEL,PERSONDET_EXP_FILE,TrackerArgs,POSE_MODEL_PATH
    from cmsct.reader import VideoReader
    from cmsct.pose import draw_skeleton
    from yolox.tracker.byte_tracker import BYTETracker
    from argparse import Namespace 
    from yolox.utils.visualize import plot_tracking
    import os
    from ultralytics import YOLO
    
    save_dir="data/"
    os.makedirs(save_dir,exist_ok=True)
    video_path = "dependencies/ByteTrack/videos/palace.mp4"
    reader = VideoReader(video_path, chunk_size=60)
    detector=Detector(PERSONDET_EXP_FILE,PERSONDET_TRT_MODEL)
    args=Namespace(**TrackerArgs)
    tracker=BYTETracker(args)
    pose_model=YOLO(POSE_MODEL_PATH) 
    
    frame_id=0
    
    while True:
        frames, done = reader.read_chunk()
        all_outputs=detector.inference_batch(frames)
        for frame,out in zip(frames,all_outputs): 
            frame_id+=1
            if out is not None:
                online_targets = tracker.update(out, [reader.height,reader.width], detector.exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                online_img=plot_tracking(frame,online_tlwhs,online_ids,frame_id=frame_id)
            else:
                online_img=np.copy(frame)

            # Run pose estimation
            pose_results = pose_model.track(frame, persist=True, verbose=False)
            if pose_results[0].boxes.id is not None:
                keypoints = pose_results[0].keypoints.xy.cpu().numpy()
                # Process each detection
                for kpts in keypoints:
                    # Draw skeleton
                    draw_skeleton(online_img, kpts)

            cv2.imwrite(os.path.join(save_dir,f"{frame_id}.png"),online_img)
                    
        if not frames or done:
            break