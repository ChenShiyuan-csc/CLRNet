import cv2
import gdown
import os

# Google Drive 文件ID 和输出路径
file_id = '1NAj2VbUB5geZTMqzZaINPoXHLyQJElAC'  # 替换为你的文件ID
output_path = 'demo.mp4'

# 下载视频
if not os.path.exists(output_path):
    print("⬇️ 开始从 Google Drive 下载视频...")
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)
else:
    print("✅ 已存在本地视频，跳过下载。")

# 打开视频
cap = cv2.VideoCapture(output_path)
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps) if fps > 0 else 33
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎞 视频帧率: {fps:.2f} FPS，帧数: {total_frames}")

# 播放控制变量
paused = False
frame_idx = 0

print("▶️ 按 [space] 暂停/继续，按 [q] 退出播放")

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        text = f"Frame {frame_idx}/{total_frames}  |  FPS: {fps:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Demo Video", frame)

    key = cv2.waitKey(wait_time if not paused else 100) & 0xFF

    if key == ord('q'):
        print("⏹ 退出播放")
        break
    elif key == ord(' '):  # 空格暂停/继续
        paused = not paused
        print("⏸ 暂停" if paused else "▶️ 继续")

cap.release()
cv2.destroyAllWindows()
