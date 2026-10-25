"""Isolated NESO adapters; existing model implementations and artifacts stay intact."""
import argparse
import json
import pickle
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.models import probabilistic_baselines as pb
from GEFcom2014.models import strong_probabilistic_baselines as sb
from GEFcom2014.models.GAN.utils_gan_wasserstein import Generator_linear, Discriminator_wassertein
from GEFcom2014.models.legacy_baselines import sample_decoder, energy_score

LABELS = {'cvae': 'Conditional VAE', 'wgan_gp': 'Conditional WGAN-GP',
          'ddpm': 'Conditional DDPM', 'd3u': 'D3U (NESO-context adapted)',
          'treeffuser': 'Treeffuser'}


def configurations():
    common = dict(batch_size=128, patience=25, min_delta=1e-4, grad_clip=5.,
                  learning_rate=2e-4, weight_decay=1e-5, epochs=200,
                  day_batch_size=4, n_scenarios=100)
    return {
        'cvae': dict(common, latent_dim=20, hidden_dim=200, layers=1,
                     learning_rate=10**-3.4, weight_decay=10**-3.4, patience=30),
        'wgan_gp': dict(common, latent_dim=64, hidden_dim=256, layers=2,
                        epochs=300, weight_decay=1e-4, patience=30,
                        n_critic=5, lambda_gp=10., validation_scenarios=30),
        'ddpm': dict(common, hidden_dim=256, diffusion_steps=100, inference_steps=100),
        'd3u': dict(common, anchor_epochs=100, anchor_hidden_dim=256, anchor_layers=2,
                    anchor_learning_rate=1e-3, anchor_patience=15, hidden_dim=128,
                    patch_size=4, depth=3, heads=4, time_dim=128,
                    diffusion_steps=100, inference_steps=50),
        'treeffuser': dict(common, tree_n_repeats=5, tree_n_estimators=500,
                           tree_early_stopping_rounds=50, tree_eval_percent=.1,
                           tree_num_leaves=31, tree_learning_rate=.1, tree_n_jobs=4,
                           tree_sde='vesde', tree_sde_initialize_from_data=False,
                           tree_sampling_steps=50, tree_n_parallel=10, tree_verbose=0),
    }


def scale_training(raw):
    """Never pass TEST arrays to fit/selection functions, even as unused arguments."""
    xs, ys = StandardScaler().fit(raw['x_ls']), StandardScaler().fit(raw['y_ls'])
    arrays = {f'{kind}_{split}': scaler.transform(raw[f'{kind}_{split}']).astype(np.float32)
              for kind, scaler in [('x', xs), ('y', ys)] for split in ('ls', 'vs')}
    return arrays, xs, ys


class ConditionalVAE(nn.Module):
    """N(0,I) prior, diagonal Gaussian posterior and Gaussian observation decoder."""
    def __init__(self, context_dim, data_dim, latent_dim=20, hidden_dim=200, layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        def net(input_dim, output_dim):
            modules = []
            for _ in range(layers):
                modules += [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
                input_dim = hidden_dim
            return nn.Sequential(*modules, nn.Linear(input_dim, output_dim))
        self.encoder = net(context_dim + data_dim, 2 * latent_dim)
        self.decoder = net(context_dim + latent_dim, 2 * data_dim)

    @staticmethod
    def gaussian(parameters):
        mu, logvar = parameters.chunk(2, dim=-1)
        return mu, logvar.clamp(-8., 6.)

    def loss(self, target, context):
        mu, logvar = self.gaussian(self.encoder(torch.cat((target, context), dim=1)))
        z = mu + (.5 * logvar).exp() * torch.randn_like(mu)
        pred, obs_logvar = self.gaussian(self.decoder(torch.cat((z, context), dim=1)))
        nll = .5 * (obs_logvar + (target - pred).square() * (-obs_logvar).exp()).sum(1)
        kl = .5 * (mu.square() + logvar.exp() - 1. - logvar).sum(1)
        return (nll + kl).mean()


class LinearCritic(Discriminator_wassertein):
    """Standard WGAN critic: unrestricted output; one interpolation scalar per case."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dis = nn.Sequential(*list(self.dis.children())[:-1])

    def grad_pen(self, real, samples, context):
        epsilon = torch.rand((len(real), 1), device=real.device)
        mixed = (real.detach() * epsilon + samples.detach() * (1 - epsilon)).requires_grad_(True)
        score = self(mixed, context)
        gradient = torch.autograd.grad(score, mixed, torch.ones_like(score), create_graph=True)[0]
        return (gradient.norm(2, dim=1) - 1).square().mean()


def fit_wgan(arrays, args, device):
    common = dict(latent_s=args.latent_dim, cond_in=arrays['x_ls'].shape[1],
                  in_size=arrays['y_ls'].shape[1], gen_w=args.hidden_dim, gen_l=args.layers,
                  lambda_gp=args.lambda_gp, gpu=device.type == 'cuda')
    generator, critic = Generator_linear(**common).to(device), LinearCritic(**common).to(device)
    optimizers = [torch.optim.Adam(m.parameters(), lr=args.learning_rate, betas=(0., .9),
                                  weight_decay=args.weight_decay) for m in (generator, critic)]
    loader = pb.make_loader(arrays['x_ls'], arrays['y_ls'], args.batch_size, args.seed)
    best, state, stale, updates, history = float('inf'), None, 0, 0, []
    for epoch in range(args.epochs):
        generator.train(); critic.train()
        d_values, g_values = [], []
        for context, target in loader:
            context, target = context.to(device), target.to(device)
            critic.requires_grad_(True)
            noise = torch.randn(len(target), args.latent_dim, device=device)
            fake = generator(noise, context).detach()
            optimizers[1].zero_grad(set_to_none=True)
            loss = critic.loss(fake, target, context)
            loss.backward(); optimizers[1].step()
            d_values.append(float(loss.detach()))
            updates += 1
            if updates % args.n_critic == 0:
                critic.requires_grad_(False)
                optimizers[0].zero_grad(set_to_none=True)
                loss_g = -critic(generator(torch.randn_like(noise), context), context).mean()
                loss_g.backward(); optimizers[0].step()
                g_values.append(float(loss_g.detach()))
        samples = sample_decoder(generator, arrays['x_vs'], args.validation_scenarios,
                                 device, args.seed + 104729)
        val = energy_score(samples, arrays['y_vs'])
        if not np.isfinite(val):
            raise ValueError('Non-finite WGAN validation score')
        history.append(dict(epoch=epoch, validation_loss=val, critic_loss=float(np.mean(d_values)),
                            generator_loss=float(np.mean(g_values)) if g_values else None))
        if val < best - args.min_delta:
            best, state, stale = val, deepcopy(generator.state_dict()), 0
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f'WGAN epoch {epoch+1} VS Energy {val:.6f}', flush=True)
        if stale >= args.patience:
            break
    if state is None:
        raise RuntimeError('No WGAN checkpoint')
    generator.load_state_dict(state)
    return generator, history, sum(p.numel() for p in critic.parameters())


def sample_cvae(model, context, count, device, seed, batch_size):
    rng = torch.Generator(device=device).manual_seed(seed)
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(context), batch_size):
            x = torch.as_tensor(context[start:start+batch_size], device=device)
            expanded = x.repeat_interleave(count, 0)
            z = torch.randn(len(expanded), model.latent_dim, generator=rng, device=device)
            mu, logvar = model.gaussian(model.decoder(torch.cat((z, expanded), dim=1)))
            noise = torch.randn(mu.shape, generator=rng, device=device)
            samples = mu + (.5 * logvar).exp() * noise
            chunks.append(samples.reshape(len(x), count, -1).cpu().numpy())
    return np.concatenate(chunks)


def sample_ddpm(model, scheduler, context, data_dim, args, device, seed):
    """Explicit generation seed; the legacy sampler always uses seed zero internally."""
    rng = torch.Generator(device=device).manual_seed(seed)
    scheduler.set_timesteps(args.inference_steps, device=device)
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(context), args.day_batch_size):
            x = torch.as_tensor(context[start:start+args.day_batch_size], device=device)
            expanded = x.repeat_interleave(args.n_scenarios, 0)
            values = torch.randn(len(expanded), data_dim, generator=rng, device=device)
            for t in scheduler.timesteps:
                pred = model(values, t.expand(len(values)), expanded)
                values = scheduler.step(pred, t, values, generator=rng).prev_sample
            chunks.append(values.reshape(len(x), args.n_scenarios, data_dim).cpu().numpy())
    return np.concatenate(chunks)


def train_task(config, bundle, output, seed, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(**config, seed=seed)
    if smoke:
        args.epochs = 1
        args.n_scenarios = 5
        if args.model == 'd3u': args.anchor_epochs = 1
        if args.model == 'treeffuser':
            args.tree_n_estimators = 10
            args.tree_n_repeats = 1
    pb.set_seed(seed)
    torch.set_num_threads(4)
    device = torch.device('cpu' if args.model == 'treeffuser' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    raw, metadata, dates = load_daily_bundle(bundle)
    if metadata.get('dataset_name') != 'neso-wind':
        raise ValueError('Wrong dataset identity')
    arrays, xs, ys = scale_training(raw)
    data_dim, context_dim = arrays['y_ls'].shape[1], arrays['x_ls'].shape[1]
    if data_dim != 24 or len(set(dates['ls']) & set(dates['vs'])) or len(set(dates['test']) & (set(dates['ls']) | set(dates['vs']))):
        raise ValueError('Invalid dimensions or overlapping dates')
    started = time.perf_counter()
    histories, audit = {}, {}
    if args.model == 'cvae':
        model = ConditionalVAE(context_dim, data_dim, args.latent_dim, args.hidden_dim, args.layers).to(device)
        loader = pb.make_loader(arrays['x_ls'], arrays['y_ls'], args.batch_size, seed)
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        histories['training'], _ = pb.fit_with_validation(model, loader, arrays, model.loss, opt, args, device)
        generate = lambda x, s: sample_cvae(model, x, args.n_scenarios, device, s, args.day_batch_size)
        audit['selection_metric'] = 'VS Gaussian negative ELBO'
        audit['method'] = 'standard normal latent prior; learned diagonal Gaussian decoder; posterior uses exp(0.5*logvariance); samples include decoder noise'
    elif args.model == 'wgan_gp':
        model, histories['training'], critic_count = fit_wgan(arrays, args, device)
        generate = lambda x, s: sample_decoder(model, x, args.n_scenarios, device, s, args.day_batch_size)
        audit.update(selection_metric='VS Energy Score (30 fixed-seed scenarios)', critic_parameters=critic_count,
                     method='conditional WGAN-GP; linear critic output; scalar trajectory interpolation')
    elif args.model == 'ddpm':
        model = pb.ConditionalDenoiser(data_dim, context_dim, args.hidden_dim).to(device)
        scheduler = pb.DDPMScheduler(num_train_timesteps=args.diffusion_steps,
                                    beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction', clip_sample=False)
        histories['training'], _ = pb.train_ddpm(model, scheduler, arrays, args, device)
        generate = lambda x, s: sample_ddpm(model, scheduler, x, data_dim, args, device, s)
        audit['selection_metric'] = 'VS fixed-noise velocity prediction MSE'
    elif args.model == 'd3u':
        conditioner = sb.DeterministicConditioner(context_dim, data_dim, args.anchor_hidden_dim, args.anchor_layers).to(device)
        histories['conditioner'] = sb.fit_conditioner(conditioner, arrays, args, device)
        model = sb.PatchDenoiser(data_dim, args.anchor_hidden_dim+data_dim, args.hidden_dim,
                                args.patch_size, args.depth, args.heads, args.time_dim).to(device)
        scheduler = pb.DDPMScheduler(num_train_timesteps=args.diffusion_steps,
                                    beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction', clip_sample=False)
        histories['training'] = sb.fit_d3u(model, conditioner, scheduler, arrays, args, device)
        generate = lambda x, s: sb.sample_d3u(model, conditioner, scheduler, x, args.n_scenarios,
                                             args.inference_steps, device, s, args.day_batch_size)
        torch.save(conditioner.state_dict(), output/'conditioner.pt')
        audit.update(selection_metric='VS point MSE then fixed-noise diffusion MSE',
                     conditioner_parameters=sum(p.numel() for p in conditioner.parameters()),
                     method='D3U adapted: supplied NESO context replaces original historical-window inputs; frozen learned conditioner and residual PatchDN diffusion')
    elif args.model == 'treeffuser':
        model = sb.build_treeffuser(args)
        model.fit(arrays['x_ls'], arrays['y_ls'])
        generate = lambda x, s: sb.sample_treeffuser(model, x, args.n_scenarios, args, s)
        audit.update(selection_metric='official LS-only internal holdout; fixed transferred hyperparameters',
                     method='official Treeffuser 0.2.0; LS internal holdout, never TEST')
    else:
        raise ValueError(args.model)
    if device.type == 'cuda': torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - started
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), output/'model.pt')
        audit['model_parameters'] = sum(p.numel() for p in model.parameters())
    else:
        pb.dump_pickle(output/'model.pickle', model)
        audit['model_parameters'] = None
    pb.dump_pickle(output/'scalers.pickle', {'context': xs, 'target': ys})
    splits = ['VS'] if smoke else ['VS', 'TEST']
    sampling_seconds = {}
    for split in splits:
        begin = time.perf_counter()
        context = xs.transform(raw['x_'+split.lower()]).astype(np.float32)
        samples = generate(context, seed + (104729 if split == 'VS' else 130363))
        if device.type == 'cuda': torch.cuda.synchronize()
        sampling_seconds[split] = time.perf_counter() - begin
        matrix = pb.scenarios_to_period_matrix(samples, ys, 'neso-wind', np.array([], dtype=int))
        if matrix.shape != (len(context)*24, args.n_scenarios) or not np.isfinite(matrix).all():
            raise ValueError('Invalid output scenarios')
        pb.dump_pickle(output/f'scenarios_{args.n_scenarios}_{split}.pickle', matrix)
        print(f'Saved {split} {matrix.shape}', flush=True)
    metadata = dict(model=args.model, label=LABELS[args.model], seed=seed, dataset_bundle=str(Path(bundle).resolve()),
                    config=vars(args), scenario_splits=splits, dates={s: dates[s.lower()].tolist() for s in splits},
                    generation_seeds={s: seed+(104729 if s=='VS' else 130363) for s in splits},
                    fit_split='LS', test_used_for_selection=False, device=str(device),
                    fit_seconds=fit_seconds, sampling_seconds=sampling_seconds, **audit, histories=histories)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    train_task(json.loads(args.config.read_text()), args.bundle, args.output, args.seed, args.smoke)
