import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#@title Defining a time-dependent score-based model (double click to expand or collapse)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianFourierProjection(nn.Module):
  # 对时间进行特定傅里叶编码,类似transform的position embedding

  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)  #
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi # 2 pi w t
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std # standard deviation of the perturbation kernel p_{0t}(x(t) | x(0)).

  def forward(self, x, t):
    # Obtain the Gaussian random feature embedding for t
    embed = self.act(self.embed(t))
    # Encoding path
    h1 = self.conv1(x)
    ## Incorporate information from t （每层卷积后注入时间信息）
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]  # rescale U-Net的输出：目的为了预测分数的二阶范数逼近真实分数的二阶范数
    return h

#@title Set up the SDE
import functools

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

def marginal_prob_std(t, sigma):
  """Compute the standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The standard deviation.
  """
  t = torch.tensor(t, device=device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t, device=device)

sigma =  25.0#@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)  # 构建无参函数：只需输入参数t
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


#@title Define the loss function (double click to expand or collapse)

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """

  # step 1： 从[0.00001,0.99999]中随机生成batch size个浮点型t
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  # step 2：基于重参数技巧，采样出分布p_t(x)的一个随机样本 perturbed_x
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  # step 3: 将加噪的x和时间点输入， 预测t时刻 对应的score
  score = model(perturbed_x, random_t)
  # step 4: 计算score matching loss
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3))) # |sigma * s + z|
  return loss


  # @title EMA

# 模型参数是通过加权回归更新的(平滑)
class EMA(nn.Module):
  def __init__(self, model, decay=0.9999, device=None):
    # model: 前一次训练的model
    super(EMA, self).__init__()
    # make a copy of model for accumulating moving average of weights
    from copy import deepcopy
    self.module = deepcopy(model)
    self.module.eval()
    self.decay = decay
    self.device = device
    if self.device is not None:
      self.module.to(device=self.device)

  def _update(self, model, update_fn):
    # 更新后的model
    with torch.no_grad():
      for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
        if self.device is not None:
          model_v.to(device=self.device)
        ema_v.copy_(update_fn(ema_v, model_v))
  def update(self, model):
    self._update(model, update_fn=lambda e, m: self.decay * e + (1-self.decay) + m)  # 更新模型：decay * 旧参数 + (1-decay) * 新参数
  def set(self, model):
    self._update(model, update_fn=lambda e, m: m)  # 直接更新模型为最新参数


import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm


score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

n_epochs =   50 #@param {'type':'integer'} # 不到1小时训练完
## size of a mini-batch
batch_size =  32 #@param {'type':'integer'}
## learning rate
lr=1e-4 #@param {'type':'number'}

dataset = MNIST('/GPUFS/sysu_jjzhang_1/hzw/academicCode/diiffussionModel', train=True, transform=transforms.ToTensor(), download=True)  # 60000张训练集
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

optimizer = Adam(score_model.parameters(), lr=lr)
tqdm_epoch = range(n_epochs)
ema = EMA(score_model)
train_flag = False
if train_flag:
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = x.to(device)
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(score_model)
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        # Print the averaged training loss so far.
        #   tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        print('Average Loss: {:5f}'.format(avg_loss / num_items))

        # Update the checkpoint after each epoch of training.
        if epoch%10==0:
            torch.save(score_model.state_dict(), f'ckpt_{epoch}_{avg_loss / num_items}.pth')


num_steps =  500#@param {'type':'integer'}
def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std, 
                           diffusion_coeff, # 扩散系数
                           batch_size=64,
                           num_steps=num_steps,
                           device='cuda',
                           eps=1e-3):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

  Returns:
    Samples.
  """
  # Step 1：定义初始时间t=1和先验分布的随机样本
  t = torch.ones(batch_size, device=device) # 生成初始时间，t=1
  init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
  # Step 2：定义采样的逆时间网格以及时间步长
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  # Step 3：根据欧拉算法求解逆时间SDE
  x = init_x
  with torch.no_grad():
    for time_step in time_steps:
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
  # step 4：取最后一步的期望作为生成的样本
  # Do not include any noise in the last sampling step.
  return mean_x

from torchvision.utils import make_grid
import time

def run_sampler(sampler, sample_batch_size = 64,figure_index=0):
    # sample_batch_size = 64 #@param {'type':'integer'}
    # sampler = ode_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

    ## Load the pre-trained checkpoint from disk.
    device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
    ckpt = torch.load('ckpt_40_17.49593082733154.pth', map_location=device)
    score_model.load_state_dict(ckpt)

    sample_batch_size = sample_batch_size
    sampler = sampler

    t1 = time.time()
    ## Generate samples using the specified sampler.
    samples = sampler(score_model,
                    marginal_prob_std_fn,
                    diffusion_coeff_fn,
                    sample_batch_size,
                    device=device)
    t2 = time.time()
    print(f"{str(sampler)}采样耗时{t2-t1} s")

    ## Sample visualization.
    samples = samples.clamp(0.0, 1.0)
    # %matplotlib inline
    import matplotlib.pyplot as plt
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    # plt.show()
    plt.savefig("/GPUFS/sysu_jjzhang_1/hzw/academicCode/diiffussionModel"+'/test_{}.png'.format(figure_index))
    plt.savefig("/GPUFS/sysu_jjzhang_1/hzw/academicCode/diiffussionModel"+'/pdf_{}.png'.format(figure_index))

sampler = Euler_Maruyama_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}
run_sampler(sampler)