PERSONDET_TRT_MODEL="dependencies/ByteTrack/pretrained/yolox_x_mix_det/model_trt.pth"
PERSONDET_EXP_FILE ="dependencies/ByteTrack/exps/example/mot/yolox_x_mix_det.py"

TrackerArgs={
    "aspect_ratio_thresh" : 1.6,
    "min_box_area" : 10,
    "track_thresh":0.5,
    "track_buffer":30,
    "match_thresh":0.8,
    "mot20":False
}