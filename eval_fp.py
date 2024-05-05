import torch
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

import sde_lib
import datasets
import sampling


from models import ddpm
from models import utils as mutils
import configs.default_cifar10_configs
import configs.vp.ddpm.cifar10_continuous


def init_wandb(config, num_samples):
    wandb.init(
        name=f'fp-sony-{num_samples}',
        # set the wandb project where this run will be logged
        project='kinetic-fp',
        # name= get_run_name(config),
        tags= ['cifar','sony','fp-quality'],
        # # track hyperparameters and run metadata
        config=config
    )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
config = configs.vp.ddpm.cifar10_continuous.get_config()
# print(config)
N = 250
sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=N)
sampling_eps = 1e-3
score_model = mutils.create_model(config)
loaded_dict  = torch.load('checkpoints/ddpm_continuous.pth')["model"]
score_model.load_state_dict(loaded_dict, strict=False)
real_score_model = mutils.get_score_fn(sde,score_model,False,True)

num_params = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
print("Number of parameters in the network:", num_params)

config.eval.batch_size = 128
num_batches_fp = 50
train_ds, eval_ds, _ = datasets.get_dataset(config,
                                            uniform_dequantization=config.data.uniform_dequantization,
                                            evaluation=True)

init_wandb(config, num_batches_fp * config.eval.batch_size)

def sample():
    sampling_shape = (32, config.data.num_channels,
                    config.data.image_size, config.data.image_size)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    samples, n = sampling_fn(score_model)
    return samples.permute(0,2,3,1)

def eval_fp():
    n_time = 50
    ts = torch.linspace(sampling_eps,1,n_time,device=device)

    fp_loss = []
    PDE = sde.PDE(real_score_model)
    PDE.train = False 
    fp_loss_fn = PDE.fp
    n_batches = num_batches_fp
    eval_iter = iter(eval_ds)
    for _ in tqdm(range(n_time)):
        t = ts[_].unsqueeze(-1)
        cur_fp = 0
        n_items = 0
        for i in tqdm(range(n_batches),leave=False):
            try:
                a = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_ds)
                a = next(eval_iter)
            x = a['image'].numpy()
            x = torch.tensor(x,device=device).permute(0,3,1,2)
            mean, std = sde.marginal_prob(x,t)
            xt = mean + torch.randn_like(mean) * std
            cur_fp += fp_loss_fn(xt,t) * xt.shape[0]
            n_items += xt.shape[0]
            # print(cur_fp)
        cur_fp = cur_fp/n_items
        fp_loss.append(cur_fp)
        wandb.log({'fp_loss_t': cur_fp, 't':t})
    
    return ts, fp_loss

def visualize(x):
    n_time = 10
    ts = torch.linspace(sampling_eps,1,n_time,device=device)
    for t_idx in tqdm(range(n_time)):
        t = ts[t_idx].unsqueeze(-1)
        mean, std = sde.marginal_prob(x,t)
        xt = mean + torch.randn_like(mean) * std
        xt = xt.clip(0,1)

        fig, ax = plt.subplots(4,8)
        for i in range(4):
            for j in range(8):
                ax[i,j].axis('off')
                ax[i,j].imshow(xt[4*i+j].cpu().detach().numpy())
        fig.savefig(f'images/cifar_{t[0].detach().cpu().numpy() : .3f}.png')

# samples = sample()
# visualize(samples)
eval_fp()

wandb.finish()