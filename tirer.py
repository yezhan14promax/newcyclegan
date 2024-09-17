import cv2
import os

def extract_frames(video_paths, output_dir, save_interval=10):
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for video_path in video_paths:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 遍历视频中的每一帧
        for i in range(frame_count):
            ret, frame = cap.read()
            if ret:
                # 仅在帧编号是 save_interval 的倍数时保存帧
                if i % save_interval == 0:
                    # 将视频文件名和帧编号组合成帧文件名
                    frame_filename = f'frame_{os.path.basename(video_path).split(".")[0]}_{i:05d}.png'
                    cv2.imwrite(os.path.join(output_dir, frame_filename), frame)
            else:
                break

        # 释放视频捕获对象
        cap.release()

# 视频A和视频B的路径列表
video_a_paths = ['./video/maohelaoshu5.rmvb']
video_b_paths = ['./video/jinjidejvren5.mp4']

# 保存帧的目录
output_dir_a = 'frames_video_a'
output_dir_b = 'frames_video_b'

# 提取视频A和视频B中的帧
extract_frames(video_a_paths, output_dir_a, save_interval=36)
extract_frames(video_b_paths, output_dir_b, save_interval=36)
