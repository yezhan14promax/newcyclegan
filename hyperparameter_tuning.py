# hyperparameter_tuning.py

import optuna
import torch
import numpy as np
import random
import time
import copy
import os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from scipy.linalg import sqrtm
import optuna.visualization as vis

# Function to calculate FID
def calculate_fid(real_activations, fake_activations):
    """计算 Fréchet Inception Distance (FID)"""
    mu_real = np.mean(real_activations, axis=0)
    mu_fake = np.mean(fake_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    sigma_fake = np.cov(fake_activations, rowvar=False)

    # 计算 Fréchet 距离
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))

    # 处理可能的复数结果
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

# Load Inception V3 model
def load_inception_model():
    weights = Inception_V3_Weights.DEFAULT
    inception_model = inception_v3(weights=weights, transform_input=False)
    inception_model.fc = torch.nn.Identity()  # 移除最后的全连接层
    inception_model.eval()  # 设置为评估模式
    return inception_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Function to extract features for FID
def extract_inception_features(model, image):
    """提取图像的 InceptionV3 特征用于 FID 计算"""
    # 调整图像维度
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

    image = preprocess(image.squeeze(0)).unsqueeze(0)
    image = image.to(next(model.parameters()).device)

    with torch.no_grad():
        pred = model(image)

    return pred.cpu().numpy().flatten()

def objective(trial):
    print(f"Starting Trial {trial.number} with parameters: {trial.params}")
    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 定义要调优的超参数
    lambda_A = trial.suggest_float('lambda_A', 5.0, 15.0)
    lambda_B = trial.suggest_float('lambda_B', 5.0, 15.0)
    lambda_identity = trial.suggest_float('lambda_identity', 0.0, 0.9)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    lambda_temporal = trial.suggest_float('lambda_temporal', 0.0, 10.0)
    print(f"Hyperparameters: lambda_A={lambda_A}, lambda_B={lambda_B}, ...")
    
    # 创建训练选项的副本并更新超参数
    opt_t = copy.deepcopy(opt)
    opt_t.lambda_A = lambda_A
    opt_t.lambda_B = lambda_B
    opt_t.lambda_identity = lambda_identity
    opt_t.lr = learning_rate
    opt_t.lambda_temporal = lambda_temporal

    # 创建模型
    model = create_model(opt_t)
    model.setup(opt_t)

    # 限制训练步骤
    max_epochs = 10  # 限制训练 5 个 epoch
    max_steps_per_epoch = 200  # 每个 epoch 限制训练 50 个步骤

    # 加载训练数据
    dataset = create_dataset(opt_t)

    # 加载验证数据
    val_opt = copy.deepcopy(opt_t)
    val_opt.phase = 'test'
    val_opt.no_flip = True
    val_dataset = create_dataset(val_opt)

    # 加载 InceptionV3 模型用于 FID 计算
    inception_model = load_inception_model()

    # 训练循环
    global_step = 0  # 用于剪枝的全局步骤计数
    for epoch in range(1, max_epochs + 1):
        for i, data in enumerate(dataset):
            if i >= max_steps_per_epoch:
                break
            global_step += 1
            model.set_input(data)
            model.optimize_parameters()

            # 每隔一定步骤执行一次剪枝检查
            if global_step % 10 == 0:
                # 验证模型
                print(f"Trial {trial.number}, Global Step {global_step}")
                real_activations = []
                fake_activations = []

                for j, val_data in enumerate(val_dataset):
                    if j >= 10:  # 只验证 10 个样本
                        break
                    model.set_input(val_data)
                    model.test()
                    real_feat = extract_inception_features(inception_model, model.real_A)
                    fake_feat = extract_inception_features(inception_model, model.fake_B)
                    real_activations.append(real_feat)
                    fake_activations.append(fake_feat)

                # 计算 FID
                fid_value = calculate_fid(np.array(real_activations), np.array(fake_activations))

                # 向 Optuna 报告中间结果
                trial.report(fid_value, global_step)

                # 检查是否应当剪枝
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

    # 最终验证
    real_activations = []
    fake_activations = []

    for val_data in val_dataset:
        model.set_input(val_data)
        model.test()
        real_feat = extract_inception_features(inception_model, model.real_A)
        fake_feat = extract_inception_features(inception_model, model.fake_B)
        real_activations.append(real_feat)
        fake_activations.append(fake_feat)

    # 计算最终的 FID
    fid_value = calculate_fid(np.array(real_activations), np.array(fake_activations))

    return fid_value

if __name__ == '__main__':
    # 解析训练选项
    opt = TrainOptions().parse()

    # 创建保存结果的文件夹
    result_dir = 'hyperparameter'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 创建 Optuna Study 对象，使用 Median 剪枝策略
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.NopPruner()
    )

    # 开始优化，使用多核 CPU，加速优化过程
    study.optimize(objective, n_trials=40, n_jobs=1)  # 根据您的 CPU 核心数调整 n_jobs

    # 保存最佳超参数结果到指定文件夹
    best_trial = study.best_trial
    result_file = os.path.join(result_dir, 'best_hyperparameters.txt')
    with open(result_file, 'w') as f:
        f.write('Best trial:\n')
        f.write('  FID score: {}\n'.format(best_trial.value))
        f.write('  Params:\n')
        for key, value in best_trial.params.items():
            f.write('    {}: {}\n'.format(key, value))

    print(f'最佳超参数已保存到 {result_file}')

    # 使用可视化功能并保存图表到指定文件夹
    # 优化历史图
    fig1 = vis.plot_optimization_history(study)
    fig1_file = os.path.join(result_dir, 'optimization_history.png')
    fig1_html = os.path.join(result_dir, 'optimization_history.html')
    fig1.write_image(fig1_file)
    fig1.write_html(fig1_html)

    # 超参数重要性图
    fig2 = vis.plot_param_importances(study)
    fig2_file = os.path.join(result_dir, 'param_importances.png')
    fig2_html = os.path.join(result_dir, 'param_importances.html')
    fig2.write_image(fig2_file)
    fig2.write_html(fig2_html)

    print(f"图表已保存到 '{result_dir}' 文件夹中")
