from pathlib import Path
import os
import cv2
import glob
from tqdm import tqdm
import numpy as np
import torch
from shapely.geometry import LineString, box
from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes
from clrnet.utils.net_utils import load_network
from ultralytics import YOLO


class YoloDetect(object):
    def __init__(self):
        self.model = YOLO("yolo11m.pt")
        
    def inference(self, img_path):
        results = self.model.predict(source=img_path, classes=[2, 5], verbose=False)
        return results
        
class LaneDetect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path':img_path, 'ori_img':ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.heads.get_lanes(data)
        return data

    def show(self, data):
        out_file = self.cfg.savedir
        if out_file:
            out_file = os.path.join(out_file, os.path.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        imshow_lanes(data['ori_img'], lanes, show=self.cfg.show, out_file=out_file)


    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        if self.cfg.show or self.cfg.savedir:
            self.show(data)
        return data

def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths 

def process(args):
    cfg = Config.fromfile(args.config)
    cfg.show = args.show
    cfg.load_from = args.load_from
    
    if args.img.lower().endswith(('.mp4', '.avi', '.mov')):
        video_name = Path(args.img).stem
        cfg.savedir = os.path.join(args.savedir, video_name)
    else:
        cfg.savedir = args.savedir
    os.makedirs(cfg.savedir, exist_ok=True)
    
    lane_detect = LaneDetect(cfg)
    yolo_detect = YoloDetect()

    if args.img.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(args.img)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"📽 原始视频帧率: {fps:.2f}, 总帧数: {frame_count}")
        frame_idx = 0
        saved_frames = []
        
        with tqdm(total=frame_count, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                temp_path = os.path.join(cfg.savedir, f'frame_{frame_idx:05d}.jpg')
                cv2.imwrite(temp_path, frame)
                
                lane_detections = lane_detect.run(temp_path)
                lanes = [lane.to_array(lane_detect.cfg) for lane in lane_detections['lanes']]
                lane_sets = [ [tuple(point) for point in lane] for lane in lanes ]
                
                yolo_detections = yolo_detect.inference(temp_path)
                yolo_boxes = yolo_detections[0].boxes.xyxy.cpu().numpy()  # Get the bounding box coordinates
                
                press_detect(lane_sets, yolo_boxes, temp_path, cfg.savedir)
                saved_frames.append(temp_path)
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        if args.video_out:
            input_video_name = Path(args.img).stem
            output_video_name = f"{input_video_name}_processed.mp4"
            out_video = os.path.join(cfg.savedir, output_video_name)

            imgs = sorted(saved_frames)
            first_frame = cv2.imread(imgs[0])
            height, width, _ = first_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
            for f in imgs:
                frame = cv2.imread(f)
                writer.write(frame)
            writer.release()
            print(f"✅ 合成视频保存至: {out_video}")

            for f in imgs:
                os.remove(f)
            print("🧹 临时帧图像已清理")

    
    else:
        paths = get_img_paths(args.img)
        for p in tqdm(paths):
            lane_detections = lane_detect.run(p)
            lanes = [lane.to_array(lane_detect.cfg) for lane in lane_detections['lanes']]
            lane_sets = [ [tuple(point) for point in lane] for lane in lanes ]
            yolo_detections = yolo_detect.inference(p)
            yolo_boxes = yolo_detections[0].boxes.xyxy.cpu().numpy()  # Get the bounding box coordinates
            press_detect(lane_sets, yolo_boxes, p, cfg.savedir)

def press_detect(lane_sets, vehicle_bboxes, img_path, savedir):
    # --- 读取图像 ---
    img = cv2.imread(img_path)
    center_img_x = img.shape[1] / 2
    
    buffer_width = 3
    iou_threshold = 0.1
    center_left_range = [0.1, 0.3]
    center_right_range = [0.7, 0.9]
    center_top_ratio = 0.7
    center_bottom_ratio = 0.85
    min_y_threshold = 0.5 * img.shape[0]  # 车头离地面50%高度以上的目标
    
    # --- 车道线缓冲区 ---
    
    lane_buffers = []
    for lane_points in lane_sets:
        lane_line = LineString(lane_points)
        lane_buffer = lane_line.buffer(buffer_width)
        lane_buffers.append(lane_buffer)
        
        # 可视化车道线
        for i in range(len(lane_points) - 1):
            pt1 = (int(lane_points[i][0]), int(lane_points[i][1]))
            pt2 = (int(lane_points[i + 1][0]), int(lane_points[i + 1][1]))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)

        # 可视化缓冲区
        for poly in lane_buffer.geoms if lane_buffer.geom_type == 'MultiPolygon' else [lane_buffer]:
            coords = np.array(poly.exterior.coords, np.int32)
            cv2.polylines(img, [coords], isClosed=True, color=(0, 255, 255), thickness=1)
    
    # --- 遍历每辆车，检查是否压线 ---
    for bbox in vehicle_bboxes:
        x_min, y_min, x_max, y_max = bbox
        ceter_box_x = (x_min + x_max) / 2
        offset_x = abs(center_img_x - ceter_box_x) / center_img_x
        center_left_ratio = center_left_range[0] + offset_x * (center_left_range[1] - center_left_range[0])
        center_right_ratio = center_right_range[1] - offset_x * (center_right_range[1] - center_right_range[0])
        
        # 过滤掉疑似车头（离镜头最近）的目标
        if y_min > min_y_threshold:
            continue
        if x_max - x_min > 0.5 * img.shape[1] or y_max - y_min > 0.5 * img.shape[0]:
            continue
        
        h = y_max - y_min
        w = x_max - x_min
        vehicle_center_x = (x_min + x_max) / 2

        # 构造底部线和底部区域
        bottom_line = LineString([(x_min, y_max), (x_max, y_max)])
        center_top = y_min + h * center_top_ratio
        center_bottom = y_min + h * center_bottom_ratio
        center_left = x_min + w * center_left_ratio
        center_right = x_min + w * center_right_ratio
        center_box = box(center_left, center_top, center_right, center_bottom)
        # center_box = box(x_min, center_top, x_max, center_bottom)

        is_violation = False
        max_iou = 0.0

        for lane_buffer in lane_buffers:
            if lane_buffer.intersects(bottom_line):
                intersection_area = lane_buffer.intersection(center_box).area
                iou = intersection_area / center_box.area if center_box.area > 0 else 0.0
                if iou > iou_threshold:
                    is_violation = True
                    max_iou = max(max_iou, iou)
                    break  # 一个车压一条线即为压线
        
        # 可视化 bbox 和底部区域
        color = (0, 0, 255) if is_violation else (0, 255, 0)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        cv2.rectangle(img, (int(center_left), int(center_top)), (int(center_right), int(center_bottom)), (200, 200, 200), 1)
        label = f"Violation ({max_iou:.3f})" if is_violation else f"OK ({max_iou:.3f})"
        cv2.putText(img, label, (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    img_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(savedir, img_name), img)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img', help='Path to image/video input')
    parser.add_argument('--show', action='store_true', help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default='output', help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    parser.add_argument('--video_out', action='store_true', help='Whether to generate video from frames and clean temp images')
    args = parser.parse_args()
    process(args)