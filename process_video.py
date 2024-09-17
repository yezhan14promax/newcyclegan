import os
import cv2
import torch
import numpy as np
import csv
from models import create_model
from util import util
from moviepy.editor import ImageSequenceClip, AudioFileClip
from options.base_options import BaseOptions
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from datetime import datetime
from scipy.linalg import sqrtm
from torchvision.models import inception_v3

class CustomOptions:
    def __init__(self, model_folder):
        self.dataroot = './datasets'  # 虚拟路径
        self.name = 'my_project4'
        self.model = 'cycle_gan'
        self.phase = 'test'
        self.checkpoints_dir = model_folder
        self.num_threads = 0
        self.batch_size = 1
        self.serial_batches = True
        self.no_flip = True
        self.display_id = -1
        self.eval = True
        self.isTrain = False  # 表示这是测试模式
        self.gpu_ids = [0] if torch.cuda.is_available() else []  # 自动检测GPU
        self.preprocess = 'resize_and_crop'  # 添加 preprocess 属性，通常与训练设置一致
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.netD = 'basic'
        self.netG = 'resnet_9blocks'
        self.n_layers_D = 3
        self.norm = 'instance'
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.no_dropout = False
        self.load_iter = 0  # 默认从最新的模型开始加载
        self.epoch = 'latest'  # 添加 epoch 属性
        self.verbose = False  # 添加 verbose 属性，控制是否打印详细的网络信息
        self.direction = 'BtoA'  # 确保方向为 BtoA 或 AtoB

def process_frame(frame, model, direction='BtoA'):
    # 将 numpy 数组转换为 PIL 图像
    frame_pil = Image.fromarray(frame)
    
    # 定义转换：将 PIL 图像转换为 PyTorch 张量
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为 [0, 1] 之间的张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1, 1]
    ])
    
    frame_tensor = transform(frame_pil).unsqueeze(0)  # 添加批次维度

    # 创建输入字典，并添加一个假的路径
    if direction == 'BtoA':
        model.set_input({'B': frame_tensor, 'A': frame_tensor, 'B_paths': '', 'A_paths': ''})
        model.test()
        visuals = model.get_current_visuals()
        processed_frame = visuals['fake_A']  # 使用 BtoA 方向时，获取 fake_A
    else:
        model.set_input({'A': frame_tensor, 'B': frame_tensor, 'A_paths': '', 'B_paths': ''})
        model.test()
        visuals = model.get_current_visuals()
        processed_frame = visuals['fake_B']  # 使用 AtoB 方向时，获取 fake_B

    # 确保处理后的图像为张量并移动到 CPU
    processed_frame = processed_frame.squeeze(0).cpu().detach().numpy()

    # 检查张量形状并进行相应的转换
    if processed_frame.shape[0] == 3:  # 检查是否为 [C, H, W]
        processed_frame_np = processed_frame.transpose(1, 2, 0)  # 转换为 [H, W, C] 格式
    else:
        processed_frame_np = processed_frame  # 如果形状已经是 [H, W, C]，不做任何转换

    # 反归一化并转换为 [0, 255] 之间的整数
    processed_frame_np = (processed_frame_np + 1) / 2.0 * 255.0
    processed_frame_np = processed_frame_np.astype(np.uint8)

    return processed_frame_np

def calculate_metrics(original_frame, generated_frame, inception_model, real_activations, real_count):
    original_frame_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    generated_frame_gray = cv2.cvtColor(generated_frame, cv2.COLOR_BGR2GRAY)

    mse = np.mean((original_frame_gray - generated_frame_gray) ** 2)
    psnr_value = psnr(original_frame_gray, generated_frame_gray)
    ssim_value = ssim(original_frame_gray, generated_frame_gray)

    # 计算生成图像的 FID 特征
    generated_activations = extract_inception_features(inception_model, generated_frame)
    
    # 更新生成图像的激活特征（累加，用于计算 FID）
    real_activations.append(generated_activations)
    real_count += 1
    
    return mse, psnr_value, ssim_value

def load_inception_model():
    # 加载 InceptionV3 模型，并设置为特征提取模式
    inception_model = inception_v3(pretrained=False, aux_logits=False, transform_input=False)
    inception_model.fc = torch.nn.Identity()  # 移除最后的全连接分类层
    inception_model = inception_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    inception_model.eval()
    return inception_model

def extract_inception_features(model, image):
    """提取图像的 InceptionV3 特征用于 FID 计算"""
    # 确保图像是 RGB 格式
    if len(image.shape) == 2 or image.shape[2] == 1:  # 如果图像是灰度图，转换为 RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 将图像转换为 PyTorch 张量，并确保是 [Batch, Channels, Height, Width] 格式
    image_tensor = transforms.ToTensor()(Image.fromarray(image)).unsqueeze(0)
    image_tensor = image_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    #print(f"Input tensor shape: {image_tensor.shape}")  # 打印输入张量的形状

    with torch.no_grad():
        pred = model(image_tensor)

    # 打印 Inception 模型输出的形状
    #print(f"Inception model output shape: {pred.shape}")

    # 直接返回特征向量（已经是 [batch_size, 2048] 的形式，无需再做池化）
    return pred.cpu().numpy().flatten()


def calculate_fid(real_activations, fake_activations):
    """计算 Fréchet Inception Distance (FID)"""
    mu_real = np.mean(real_activations, axis=0)
    mu_fake = np.mean(fake_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    sigma_fake = np.cov(fake_activations, rowvar=False)

    # 计算 Fréchet 距离
    diff = mu_real - mu_fake
    covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid


def temporal_smoothing(current_frame, previous_frame, alpha=0.05):
    return cv2.addWeighted(current_frame, alpha, previous_frame, 1 - alpha, 0)

def main(video_path, model_folder):
    # 使用自定义选项
    opt = CustomOptions(model_folder)

    # 创建模型
    model = create_model(opt)
    model.setup(opt)

    # 加载 InceptionV3 模型（用于 FID 计算）
    inception_model_path = './inception_v3/inception_v3_google-0cc3c7bd.pth'  # 替换为你保存模型的路径
    inception_model = inception_v3(pretrained=False, transform_input=False)  # 设置 pretrained=False
    inception_model.load_state_dict(torch.load(inception_model_path))  # 手动加载预训练权重
    inception_model = inception_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    inception_model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {video_path}")

    # 创建输出文件夹
    output_dir = os.path.join('./output_video', f"{os.path.splitext(os.path.basename(video_path))[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{opt.direction}")
    os.makedirs(output_dir, exist_ok=True)

    # 创建帧保存文件夹
    frame_folder = os.path.join(output_dir, 'frames')
    os.makedirs(frame_folder, exist_ok=True)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 准备记录指标的CSV文件
    metrics_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_{opt.direction}.csv"
    metrics_path = os.path.join(output_dir, metrics_filename)

    # FID 相关变量
    real_activations = []
    fake_activations = []
    real_count = 0
    fake_count = 0
    
    with open(metrics_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'MSE', 'PSNR', 'SSIM', 'FID'])

        processed_frames = []
        prev_processed_frame = None
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # 处理每一帧
            processed_frame = process_frame(frame, model, direction=opt.direction)
            
            # 如果不是第一帧，则应用时序滤波
            if prev_processed_frame is not None:
                processed_frame = temporal_smoothing(processed_frame, prev_processed_frame, alpha=0.5)

            prev_processed_frame = processed_frame
            processed_frames.append(processed_frame)

            # 计算 MSE, PSNR, SSIM 和 FID
            mse, psnr_value, ssim_value = calculate_metrics(frame, processed_frame, inception_model, real_activations, real_count)
            
            # 提取真实图像的 Inception 特征
            real_activations.append(extract_inception_features(inception_model, frame))
            real_count += 1
            
            # 提取生成图像的 Inception 特征
            fake_activations.append(extract_inception_features(inception_model, processed_frame))
            fake_count += 1

            if i % 10 == 0:
                print(f"Processed {i + 1}/{frame_count} frames")

        # 计算 FID
        fid_value = calculate_fid(np.array(real_activations), np.array(fake_activations))
        print(f"Final FID for the video: {fid_value}")

        writer.writerow([frame_count, mse, psnr_value, ssim_value, fid_value])
           
    cap.release()

    # 保存处理后的视频
    clip = ImageSequenceClip(processed_frames, fps=fps)

    # 添加原音轨
    original_audio = AudioFileClip(video_path)
    clip = clip.set_audio(original_audio)
    
    output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_{opt.direction}.mp4")
    clip.write_videofile(output_video_path, codec="libx264")

    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    video_path = './video/jvren.mp4'
    model_folder = './checkpoints'

    main(video_path, model_folder)
