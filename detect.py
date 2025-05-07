import argparse
import os
import glob
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch
from shapely.geometry import LineString, box

from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes
from clrnet.utils.net_utils import load_network
from ultralytics import YOLO


class YoloDetect:
    def __init__(self):
        self.model = YOLO("yolo11m.pt")

    def inference(self, img_path):
        return self.model.predict(source=img_path, classes=[2, 5], verbose=False)


class LaneDetect:
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.DataParallel(self.net, device_ids=range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path': img_path, 'ori_img': ori_img})
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

    def run(self, img_path):
        data = self.preprocess(img_path)
        data['lanes'] = self.inference(data)[0]
        if self.cfg.show or self.cfg.savedir:
            self.show(data)
        return data


def get_img_paths(path):
    p = str(Path(path).absolute())
    if '*' in p:
        return sorted(glob.glob(p, recursive=True))
    elif os.path.isdir(p):
        return sorted(glob.glob(os.path.join(p, '*.*')))
    elif os.path.isfile(p):
        return [p]
    else:
        raise FileNotFoundError(f'Path {p} does not exist')


def handle_video(task, video_path, savedir, cfg=None, video_out=False):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    saved_frames = []
    print(f"📽 原始视频帧率: {fps:.2f}, 总帧数: {frame_count}")
    os.makedirs(savedir, exist_ok=True)

    lane_detect = LaneDetect(cfg) if task in ['lane', 'press'] else None
    yolo_detect = YoloDetect() if task in ['yolo', 'press'] else None

    with tqdm(total=frame_count, desc=f"Processing video ({task})") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            temp_path = os.path.join(savedir, f'frame_{frame_idx:05d}.jpg')
            cv2.imwrite(temp_path, frame)

            if task == 'yolo':
                results = yolo_detect.inference(temp_path)
                img = frame.copy()
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    if is_likely_front_car(x1, y1, x2, y2, img.shape):
                        continue
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(temp_path, img)

            elif task == 'lane':
                lane_detect.run(temp_path)

            elif task == 'press':
                lane_data = lane_detect.preprocess(temp_path)
                lane_data['lanes'] = lane_detect.inference(lane_data)[0]
                lanes = [lane.to_array(cfg) for lane in lane_data['lanes']]
                lane_sets = [[tuple(pt) for pt in lane] for lane in lanes]
                yolo_data = yolo_detect.inference(temp_path)
                yolo_boxes = yolo_data[0].boxes.xyxy.cpu().numpy()
                press_detect(lane_sets, yolo_boxes, temp_path, savedir)

            saved_frames.append(temp_path)
            frame_idx += 1
            pbar.update(1)

    cap.release()

    if video_out:
        out_video = os.path.join(savedir, f"{Path(video_path).stem}_{task}_processed.mp4")
        first_frame = cv2.imread(saved_frames[0])
        height, width, _ = first_frame.shape
        writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for f in saved_frames:
            frame = cv2.imread(f)
            writer.write(frame)
        writer.release()
        print(f"✅ 合成视频保存至: {out_video}")

        for f in saved_frames:
            os.remove(f)
        print("🧹 临时帧图像已清理")


def press_detect(lane_sets, vehicle_bboxes, img_path, savedir):
    img = cv2.imread(img_path)
    center_img_x = img.shape[1] / 2

    buffer_width = 3
    iou_threshold = 0.1
    center_left_range = [0.1, 0.3]
    center_right_range = [0.7, 0.9]
    center_top_ratio = 0.7
    center_bottom_ratio = 0.85

    lane_buffers = []
    for lane_points in lane_sets:
        lane_line = LineString(lane_points)
        lane_buffer = lane_line.buffer(buffer_width)
        lane_buffers.append(lane_buffer)
        for i in range(len(lane_points) - 1):
            pt1 = (int(lane_points[i][0]), int(lane_points[i][1]))
            pt2 = (int(lane_points[i + 1][0]), int(lane_points[i + 1][1]))
            cv2.line(img, pt1, pt2, (0, 255, 255), 2)
        for poly in lane_buffer.geoms if lane_buffer.geom_type == 'MultiPolygon' else [lane_buffer]:
            coords = np.array(poly.exterior.coords, np.int32)
            # cv2.polylines(img, [coords], isClosed=True, color=(0, 255, 255), thickness=1)

    for bbox in vehicle_bboxes:
        x_min, y_min, x_max, y_max = bbox
        ceter_box_x = (x_min + x_max) / 2
        offset_x = abs(center_img_x - ceter_box_x) / center_img_x
        center_left_ratio = center_left_range[0] + offset_x * (center_left_range[1] - center_left_range[0])
        center_right_ratio = center_right_range[1] - offset_x * (center_right_range[1] - center_right_range[0])

        if is_likely_front_car(x_min, y_min, x_max, y_max, img.shape):
            continue

        h = y_max - y_min
        w = x_max - x_min
        bottom_line = LineString([(x_min, y_max), (x_max, y_max)])
        center_top = y_min + h * center_top_ratio
        center_bottom = y_min + h * center_bottom_ratio
        center_left = x_min + w * center_left_ratio
        center_right = x_min + w * center_right_ratio
        center_box = box(center_left, center_top, center_right, center_bottom)

        is_violation = False
        max_iou = 0.0

        for lane_buffer in lane_buffers:
            if lane_buffer.intersects(bottom_line):
                intersection_area = lane_buffer.intersection(center_box).area
                iou = intersection_area / center_box.area if center_box.area > 0 else 0.0
                if iou > iou_threshold:
                    is_violation = True
                    max_iou = max(max_iou, iou)
                    break

        color = (0, 0, 255) if is_violation else (0, 255, 0)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        # cv2.rectangle(img, (int(center_left), int(center_top)), (int(center_right), int(center_bottom)), (200, 200, 200), 1)
        # label = f"Violation ({max_iou:.3f})" if is_violation else f"OK ({max_iou:.3f})"
        label = "Crossing!" if is_violation else ""
        cv2.putText(img, label, (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_path = os.path.join(savedir, os.path.basename(img_path))
    cv2.imwrite(out_path, img)


def is_likely_front_car(x1, y1, x2, y2, img_shape, y_thresh_ratio=0.5, w_thresh_ratio=0.5, h_thresh_ratio=0.5):
    H, W = img_shape[:2]
    w = x2 - x1
    h = y2 - y1

    if y1 > int(y_thresh_ratio * H):
        return True
    if w > int(w_thresh_ratio * W) or h > int(h_thresh_ratio * H):
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', help='CLRNet config path (required for lane/press task)')
    parser.add_argument('--img', required=True, help='Image, video or folder path')
    parser.add_argument('--task', choices=['yolo', 'lane', 'press'], default='press', help='Task type')
    parser.add_argument('--show', action='store_true', help='Show visualization')
    parser.add_argument('--savedir', type=str, default='output', help='Save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='CLRNet model path')
    parser.add_argument('--video_out', action='store_true', help='Rebuild video from processed frames')
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    if args.img.lower().endswith(('.mp4', '.avi', '.mov')):
        cfg = Config.fromfile(args.config) if args.task in ['lane', 'press'] else None
        if cfg:
            cfg.show = args.show
            cfg.savedir = args.savedir
            cfg.load_from = args.load_from
        handle_video(args.task, args.img, args.savedir, cfg, args.video_out)

    else:
        paths = get_img_paths(args.img)

        if args.task == 'yolo':
            model = YOLO("yolo11m.pt")
            for path in tqdm(paths):
                results = model.predict(source=path, classes=[2, 5], verbose=False)
                img = cv2.imread(path)
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    if is_likely_front_car(x1, y1, x2, y2, img.shape):
                        continue
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                out_path = os.path.join(args.savedir, os.path.basename(path))
                cv2.imwrite(out_path, img)

        elif args.task == 'lane':
            cfg = Config.fromfile(args.config)
            cfg.show = args.show
            cfg.savedir = args.savedir
            cfg.load_from = args.load_from
            lane_detect = LaneDetect(cfg)
            for p in tqdm(paths):
                lane_detect.run(p)

        elif args.task == 'press':
            cfg = Config.fromfile(args.config)
            cfg.show = args.show
            cfg.savedir = args.savedir
            cfg.load_from = args.load_from
            lane_detect = LaneDetect(cfg)
            yolo_detect = YoloDetect()
            for p in tqdm(paths):
                lane_data = lane_detect.preprocess(p)
                lane_data['lanes'] = lane_detect.inference(lane_data)[0]
                lanes = [lane.to_array(cfg) for lane in lane_data['lanes']]
                lane_sets = [[tuple(pt) for pt in lane] for lane in lanes]
                yolo_data = yolo_detect.inference(p)
                yolo_boxes = yolo_data[0].boxes.xyxy.cpu().numpy()
                press_detect(lane_sets, yolo_boxes, p, args.savedir)


if __name__ == '__main__':
    main()
