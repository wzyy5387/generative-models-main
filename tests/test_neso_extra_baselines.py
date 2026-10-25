import numpy as np
import torch
from GEFcom2014.models.neso_extra_baselines import (
    ConditionalVAE, LinearCritic, scale_training, sample_cvae,
)


def test_scaling_is_ls_only_and_training_does_not_receive_test():
    raw={f'{k}_{s}':np.arange(12,dtype=np.float32).reshape(4,3)+offset
         for k in ['x','y'] for s,offset in [('ls',0),('vs',100),('test',10000)]}
    first,xs,ys=scale_training(raw)
    raw['x_test'][:]=1e9;raw['y_test'][:]=-1e9
    raw['y_vs']+=1000
    second,xs2,ys2=scale_training(raw)
    assert set(first)=={'x_ls','y_ls','x_vs','y_vs'}
    np.testing.assert_array_equal(xs.mean_,xs2.mean_)
    np.testing.assert_array_equal(ys.mean_,ys2.mean_)
    np.testing.assert_array_equal(first['y_ls'],second['y_ls'])


def test_cvae_samples_gaussian_observation_noise_and_loss_has_correct_kl():
    model=ConditionalVAE(2,3,latent_dim=2,hidden_dim=4)
    for p in model.parameters():torch.nn.init.zeros_(p)
    target=torch.ones(8,3);context=torch.zeros(8,2)
    assert torch.allclose(model.loss(target,context),torch.tensor(1.5))
    samples=sample_cvae(model,np.zeros((2,2),dtype=np.float32),4000,torch.device('cpu'),17,2)
    assert samples.shape==(2,4000,3)
    assert abs(samples.var()-1)<.05
    np.testing.assert_array_equal(samples,sample_cvae(model,np.zeros((2,2),dtype=np.float32),4000,torch.device('cpu'),17,2))
    assert not np.array_equal(samples,sample_cvae(model,np.zeros((2,2),dtype=np.float32),4000,torch.device('cpu'),18,2))


def test_wgan_critic_unrestricted_and_gradient_penalty_is_differentiable():
    critic=LinearCritic(latent_s=2,cond_in=2,in_size=3,gen_w=4,gen_l=1,lambda_gp=10.,gpu=False)
    assert isinstance(critic.dis[-1],torch.nn.Linear)
    with torch.no_grad():
        critic.dis[-1].weight.zero_();critic.dis[-1].bias.fill_(-2)
    assert (critic(torch.zeros(5,3),torch.zeros(5,2))<0).all()
    penalty=critic.grad_pen(torch.randn(5,3),torch.randn(5,3),torch.randn(5,2))
    assert torch.isfinite(penalty)
    penalty.backward()
