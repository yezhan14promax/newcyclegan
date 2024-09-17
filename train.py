"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import numpy as np
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from torchvision.models import inception_v3
from util import util
from scipy.linalg import sqrtm
from torchvision import transforms
import copy

# Function to calculate FID
def calculate_fid(real_activations, fake_activations):
    """计算 Fréchet Inception Distance (FID)"""
    mu_real = np.mean(real_activations, axis=0)
    mu_fake = np.mean(fake_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    sigma_fake = np.cov(fake_activations, rowvar=False)

    # 计算 Fréchet 距离
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))  # 使用 scipy 的 sqrtm

    if np.iscomplexobj(covmean):  # 如果结果是复数，取实部
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

# Load Inception V3 model
def load_inception_model():
    inception_model = inception_v3(pretrained=False, transform_input=False)
    inception_model_path = './inception_v3/inception_v3_google-0cc3c7bd.pth'  # 使用相对路径加载模型
    inception_model.load_state_dict(torch.load(inception_model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    inception_model.fc = torch.nn.Identity()  # Remove the final fully connected layer
    inception_model.eval()  # Set to evaluation mode
    return inception_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Function to extract features for FID
def extract_inception_features(model, image):
    """提取图像的 InceptionV3 特征用于 FID 计算"""
    # 如果图像有 5 个维度，移除多余的维度
    if image.dim() == 5 and image.size(0) == 1:
        image = image.squeeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)

    # 将图像从 [-1, 1] 归一化到 [0, 1]
    image = (image + 1) / 2.0

    # 应用预处理
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = preprocess(image.squeeze(0)).unsqueeze(0)  # 注意需要先去掉批次维度，再恢复
    image = image.to(next(model.parameters()).device)

    with torch.no_grad():
        pred = model(image)

    return pred.cpu().numpy().flatten()


if __name__ == '__main__':
    opt = TrainOptions().parse()   # 获取训练选项
    print(f"Dropout enabled: {not opt.no_dropout}")
    print(f"lambda_A: {opt.lambda_A}")
    print(f"lambda_B: {opt.lambda_B}")
    print(f"lambda_identity: {opt.lambda_identity}")
    print(f"lambda_temporal: {opt.lambda_temporal}")  # 新增行，确保打印时间一致性损失的权重

    # 加载训练集数据
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'Total dataset size: {dataset_size}')

    # 创建模型
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    # Early stopping variables
    best_fid = float('inf')
    patience_counter = 0
    patience = 5

    # Load InceptionV3 model for FID calculation
    inception_model = load_inception_model()

    # 加载测试集作为验证集
    val_opt = copy.deepcopy(opt)
    val_opt.phase = 'test'
    val_opt.no_flip = True  # 验证时不进行随机翻转
    val_dataset = create_dataset(val_opt)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # 计时器
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()

        # 训练循环
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, 0)

        # 验证循环，使用测试集
        real_activations = []
        fake_activations = []

        for val_data in val_dataset.load_data():
            model.set_input(val_data)
            model.test()
            
            real_A = model.real_A  # 去掉多余的 batch 维度
            fake_B = model.fake_B  # 去掉多余的 batch 维度
                       
            real_activations.append(extract_inception_features(inception_model, real_A))
            fake_activations.append(extract_inception_features(inception_model, fake_B))    

        # 计算 FID
        fid_value = calculate_fid(np.array(real_activations), np.array(fake_activations))
        print(f'Epoch [{epoch}], FID (Validation): {fid_value}')

        # Early stopping based on FID
        if fid_value < best_fid:
            print(f"New best FID: {fid_value}, saving model...")
            best_fid = fid_value
            patience_counter = 0
            model.save_networks('best')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered.')
                break

        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving the model at the end of epoch {epoch}, iters {total_iters}')
            model.save_networks('latest')

        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay}, Time Taken: {time.time() - epoch_start_time} sec')
