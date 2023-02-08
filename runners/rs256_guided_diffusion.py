import os
import numpy as np
from tqdm import tqdm

import torch
import torchvision.utils as tvu
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.diffusion_new import ConditionalModel as CModel
from models.diffusion_new import Model
from functions.process_data import *
from functions.denoising_step import guided_ddpm_steps, guided_ddim_steps, ddpm_steps, ddim_steps

import matplotlib.pyplot as plt
from einops import rearrange
from mpl_toolkits.axes_grid1 import ImageGrid
import math
import pickle

from copy import deepcopy


class MetricLogger(object):
    def __init__(self, metric_fn_dict):
        self.metric_fn_dict = metric_fn_dict
        self.metric_dict = {}
        self.reset()

    def reset(self):
        for key in self.metric_fn_dict.keys():
            self.metric_dict[key] = []

    @torch.no_grad()
    def update(self, **kwargs):
        for key in self.metric_fn_dict.keys():
            self.metric_dict[key].append(self.metric_fn_dict[key](**kwargs))

    def get(self):
        return self.metric_dict.copy()

    def log(self, outdir, postfix=''):
        with open(os.path.join(outdir, f'metric_log_{postfix}.pkl'), 'wb') as f:
            pickle.dump(self.metric_dict, f)


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def load_flow_data(path, stat_path=None):
    # load flow data from path
    data = np.load(path)   # [N, T, h, w]

    print('Original data shape:', data.shape)
    data_mean, data_scale = np.mean(data[:-4]), np.std(data[:-4])
    print(f'Data range: mean: {data_mean} scale: {data_scale}')
    data = data[-4:, ...].copy().astype(np.float32)   # only take the test set
    data = torch.as_tensor(data, dtype=torch.float32)
    flattened_data = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]-2):
            flattened_data.append(data[i, j:j+3, ...])
    flattened_data = torch.stack(flattened_data, dim=0)
    print(f'data shape: {flattened_data.shape}')
    return flattened_data, data_mean.item(), data_scale.item()


def load_recons_data(ref_path, sample_path, data_kw, smoothing, smoothing_scale):
    with np.load(sample_path, allow_pickle=True) as f:
        sampled_data = f[data_kw][-4:, ...].copy().astype(np.float32)
        # idx_lst = f[idx_kw][-4:]
    sampled_data = torch.as_tensor(sampled_data, dtype=torch.float32)
    ref_data = np.load(ref_path).astype(np.float32)
    data_mean, data_scale = np.mean(ref_data[:-4]), np.std(ref_data[:-4])
    ref_data = ref_data[-4:, ...].copy().astype(np.float32)   # only take the test set
    ref_data = torch.as_tensor(ref_data, dtype=torch.float32)

    flattened_sampled_data = []
    flattened_ref_data = []

    for i in range(ref_data.shape[0]):
        for j in range(ref_data.shape[1] - 2):
            flattened_ref_data.append(ref_data[i, j:j + 3, ...])
            flattened_sampled_data.append(sampled_data[i, j:j + 3, ...])
            # mask_lst.append(mask)

    flattened_ref_data = torch.stack(flattened_ref_data, dim=0)
    flattened_sampled_data = torch.stack(flattened_sampled_data, dim=0)
    if smoothing:
        arr = flattened_sampled_data
        ker_size = smoothing_scale
        # peridoic padding
        arr = F.pad(arr,
                    pad=((ker_size - 1) // 2, (ker_size - 1) // 2, (ker_size - 1) // 2, (ker_size - 1) // 2),
                    mode='circular', )
        arr = transforms.GaussianBlur(kernel_size=ker_size, sigma=ker_size)(arr)# F.avg_pool2d(arr, (ker_size, ker_size), stride=1, count_include_pad=False)
        flattened_sampled_data = arr[..., (ker_size - 1) // 2:-(ker_size - 1) // 2, (ker_size - 1) // 2:-(ker_size - 1) // 2]

    print(f'data shape: {flattened_ref_data.shape}')

    return flattened_ref_data, flattened_sampled_data, data_mean.item(), data_scale.item()


class MinMaxScaler(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, x):
        return (x - self.min) #/ (self.max - self.min)

    def inverse(self, x):
        return x * (self.max - self.min) + self.min

    def scale(self):
        return self.max - self.min


class StdScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def scale(self):
        return self.std


def nearest_blur_image(data, scale):
    # use pytorch's bulit-in transforms to blur the data
    blur_data = data[:, :, ::scale, ::scale]

    return blur_data


def gaussian_blur_image(data, scale):
    # use pytorch's bulit-in transforms to blur the data
    blur_data = transforms.GaussianBlur(kernel_size=scale, sigma=2*scale+1)(data)

    return blur_data


def random_square_hole_mask(data, hole_size):
    # generate a random square hole mask
    h, w = data.shape[2:]
    mask = torch.zeros(data.shape, dtype=torch.int64).to(data.device)
    hole_x = np.random.randint(0, w - hole_size)
    hole_y = np.random.randint(0, h - hole_size)
    mask[..., hole_y:hole_y+hole_size, hole_x:hole_x+hole_size] = 1

    return mask


def make_image_grid(images, out_path, ncols=10):
    # assume images in the shape of (N, T, H, W)
    t, h, w = images.shape
    images = images.detach().cpu().numpy()
    b = t // ncols
    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(b, ncols),  # creates 2x2 grid of axes
                     )

    for ax, im_no in zip(grid, np.arange(b*ncols)):
        # Iterating over the grid returns the Axes.
        ax.imshow(images[im_no, :, :], cmap='twilight', vmin=-23, vmax=23)
        ax.axis('off')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def slice2sequence(data):
    data = rearrange(data[:, 1:2], 't f h w -> (t f) h w')
    return data


def l1_loss(x, y):
    return torch.mean(torch.abs(x - y))


def l2_loss(x, y):
    return ((x - y)**2).mean((-1, -2)).sqrt().mean()


def voriticity_residual(w, re=1000.0, dt=1/32, calc_grad=True):
    # w [b t h w]
    batchsize = w.size(0)
    w = w.clone()
    w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)
    device = w.device

    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u*wx + v*wy

    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    # establish forcing term
    x = torch.linspace(0, 2*np.pi, nx + 1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x)
    f = -4*torch.cos(4*Y)

    residual = wt + (advection - (1.0 / re) * wlap + 0.1*w[:, 1:-1]) - f
    residual_loss = (residual**2).mean()
    if calc_grad:
        dw = torch.autograd.grad(residual_loss, w)[0]
        return dw, residual_loss
    else:
        return residual_loss


class Diffusion(object):
    def __init__(self, args, config, logger, log_dir, device=None):
        self.args = args
        self.config = config
        self.logger = logger
        self.image_sample_dir = log_dir

        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def log(self, info):
        self.logger.info(info)

    def reconstruct(self):
        self.log('Doing sparse reconstruction task')
        self.log("Loading model")

        if self.config.model.type == 'conditional':
            print('Using conditional model')
            model = CModel(self.config)
        else:
            print('Using unconditional model')
            model = Model(self.config)

        model.load_state_dict(torch.load(self.config.model.ckpt_path)[-1])

        model.to(self.device)

        self.log("Model loaded")

        model.eval()
        self.log('Preparing data')
        ref_data, blur_data, data_mean, data_std = load_recons_data(self.config.data.data_dir,
                                                                    self.config.data.sample_data_dir,
                                                                    self.config.data.data_kw,
                                                                    smoothing=self.config.data.smoothing,
                                                                    smoothing_scale=self.config.data.smoothing_scale)

        scaler = StdScaler(data_mean, data_std)
        # minmax_scaler = MinMaxScaler(ref_data.min(), ref_data.max())

        self.log("Start sampling")

        # pack data loader
        testset = torch.utils.data.TensorDataset(blur_data, ref_data)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=self.config.sampling.batch_size,
                                                  shuffle=False, num_workers=self.config.data.num_workers)

        l2_loss_all = np.zeros((ref_data.shape[0], self.args.repeat_run, self.args.sample_step))
        residual_loss_all = np.zeros((ref_data.shape[0], self.args.repeat_run, self.args.sample_step))


        for batch_index, (blur_data, data) in enumerate(test_loader):
            self.log('Batch: {} / Total batch {}'.format(batch_index, len(test_loader)))
            x0 = blur_data.to(self.device)

            gt = data.to(self.device)

            self.log('Preparing reference image')
            self.log('Dumping visualization...')

            sample_folder = 'sample_batch{}'.format(batch_index)
            ensure_dir(os.path.join(self.image_sample_dir, sample_folder))

            sample_img_filename = 'input_image.png'
            path_to_dump = os.path.join(self.image_sample_dir, sample_folder, sample_img_filename)
            x0_masked = x0.clone()
            # print(torch.any(mask))
            # x0_masked[~mask] = np.nan
            make_image_grid(slice2sequence(x0_masked), path_to_dump)
            sample_img_filename = 'reference_image.png'
            path_to_dump = os.path.join(self.image_sample_dir, sample_folder, sample_img_filename)
            make_image_grid(slice2sequence(gt), path_to_dump)

            # save as array
            if self.config.sampling.dump_arr:
                np.save(os.path.join(self.image_sample_dir, sample_folder, 'input_arr.npy'),
                        slice2sequence(x0).cpu().numpy())
                np.save(os.path.join(self.image_sample_dir, sample_folder, 'reference_arr.npy'),
                        slice2sequence(data).cpu().numpy())

            # calculate initial loss
            #l1_loss_init = l1_loss(x0, gt)
            l2_loss_init = l2_loss(x0, gt)

            self.log('L2 loss init: {}'.format(l2_loss_init))
            gt_residual = voriticity_residual(gt)[1].detach()
            init_residual = voriticity_residual(x0)[1].detach()
            self.log('Residual init: {}'.format(init_residual))
            self.log('Residual reference: {}'.format(gt_residual))

            x0 = scaler(x0)
            xinit = x0.clone()
            
            # prepare loss function
            if self.config.sampling.log_loss:
                l2_loss_fn = lambda x: l2_loss(scaler.inverse(x).to(gt.device), gt)
                equation_loss_fn = lambda x: voriticity_residual(scaler.inverse(x),
                                                                 calc_grad=False)

                logger = MetricLogger({
                    'l2 loss': l2_loss_fn,
                    'residual loss': equation_loss_fn
                })

            # we repeat the sampling for multiple times
            for repeat in range(self.args.repeat_run):
                self.log(f'Run No.{repeat}:')
                x0 = xinit.clone()
                for it in range(self.args.sample_step):  # we run the sampling for three times

                    e = torch.randn_like(x0)
                    total_noise_levels = int(self.args.t * (0.7 ** it))

                    a = (1 - self.betas).cumprod(dim=0)
                    x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                    if self.config.model.type == 'conditional':
                        physical_gradient_func = lambda x: voriticity_residual(scaler.inverse(x))[0] / scaler.scale()
                    elif self.config.sampling.lambda_ > 0:
                        physical_gradient_func = lambda x: \
                            voriticity_residual(scaler.inverse(x))[0] / scaler.scale() * self.config.sampling.lambda_

                    num_of_reverse_steps = int(self.args.reverse_steps * (0.7 ** it ))
                    betas = self.betas.to(self.device)
                    skip = total_noise_levels // num_of_reverse_steps
                    seq = range(0, total_noise_levels, skip)

                    if self.config.model.type == 'conditional':
                        xs, _ = guided_ddim_steps(x, seq, model, betas,
                                                  w=self.config.sampling.guidance_weight,
                                                  dx_func=physical_gradient_func, cache=False, logger=logger)
                    elif self.config.sampling.lambda_ > 0:
                        xs, _ = ddim_steps(x, seq, model, betas,
                                           dx_func=physical_gradient_func, cache=False, logger=logger)
                    else:
                        xs, _ = ddim_steps(x, seq, model, betas, cache=False, logger=logger)

                    x = xs[-1]
                    x0 = xs[-1].cuda()

                    l2_loss_f = l2_loss(scaler.inverse(x.clone()).to(gt.device), gt)
                    self.log('L2 loss it{}: {}'.format(it, l2_loss_f))
                    residual_loss_f = voriticity_residual(scaler.inverse(x.clone()), calc_grad=False).detach()
                    self.log('Residual it{}: {}'.format(it, residual_loss_f))

                    # l2_loss_log[f'run_{repeat}'].append(l2_loss_f.item())
                    # residual_loss_log[f'run_{repeat}'].append(residual_loss_f.item())
                    l2_loss_all[batch_index * x.shape[0]:(batch_index + 1) * x.shape[0], repeat, it] = l2_loss_f.item()
                    residual_loss_all[batch_index * x.shape[0]:(batch_index + 1) * x.shape[0], repeat,
                    it] = residual_loss_f.item()

                    if self.config.sampling.dump_arr:
                        np.save(os.path.join(self.image_sample_dir, sample_folder, f'sample_arr_run_{repeat}_it{it}.npy'),
                                slice2sequence(scaler.inverse(x)).cpu().numpy())

                    if self.config.sampling.log_loss:
                        logger.log(os.path.join(self.image_sample_dir, sample_folder), f'run_{repeat}_it{it}')
                        logger.reset()

            # with open(os.path.join(self.image_sample_dir, sample_folder, f'log.pkl'), 'wb') as f:
            #     pickle.dump({'l2 loss': l2_loss_log, 'residual loss': residual_loss_log, 'bicubic': bicubic_log}, f)
            self.log('Finished batch {}'.format(batch_index))
            self.log('========================================================')
        self.log('Finished sampling')
        self.log(f'mean l2 loss: {l2_loss_all[..., -1].mean()}')
        self.log(f'std l2 loss: {l2_loss_all[..., -1].std(axis=1).mean()}')
        self.log(f'mean residual loss: {residual_loss_all[..., -1].mean()}')
        self.log(f'std residual loss: {residual_loss_all[..., -1].std(axis=1).mean()}')



