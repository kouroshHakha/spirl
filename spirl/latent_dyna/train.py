import argparse
import torch
import os
import time
from functools import partial

from torch.utils.data import Dataset, DataLoader
from torch import autograd
from torch.optim import Adam, RMSprop, SGD

from spirl.latent_dyna.utils import MLP
from spirl.utils.general_utils import (
    AttrDict, ParamDict, map_dict, AverageMeter, get_clipped_optimizer, dummy_context
)
from spirl.components.trainer_base import BaseTrainer
from spirl.components.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path
from spirl.train import set_seeds, make_path, datetime_str, save_config, get_exp_dir, save_checkpoint
from spirl.utils.pytorch_utils import LossSpikeHook, NanGradHook, NoneGradHook, \
    DataParallelWrapper, RAdam

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import imp

from spirl.debug import register_pdb_hook
register_pdb_hook()

class DynaDataset(Dataset):

    def __init__(self, dataset: AttrDict):
        self.dataset = dataset

    def __getitem__(self, item):
        sample = {}
        for k in self.dataset.keys():
            sample[k] = self.dataset[k][item]
        return sample

    def __len__(self):
        return next(iter(self.dataset.values())).shape[0]

class DynaModel(pl.LightningModule):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None,
                 lr=1e-4,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.tloss = kwargs.get('tloss', None)
        self.vloss = kwargs.get('vloss', None)
        self.cond_skills = kwargs.get('cond_skills', True)

        self.fpass = MLP(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)


    def forward(self, inputs):
        x_in = self.get_model_input(inputs)
        delta = self.fpass(x_in)
        output = AttrDict(delta=delta)
        return output

    def get_model_input(self, inputs):
        if self.cond_skills:
            return torch.cat([inputs.skills, inputs.states], -1)
        return inputs.states

    def configure_optimizers(self):
        return Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

    def loss(self, output, inputs):
        delta_s = inputs.next_states - inputs.states
        # loss across all dims avged over samples
        loss = ((output.delta - delta_s) ** 2).sum(-1).mean(0)
        losses = AttrDict(total=loss)

        return losses

    def training_step(self, batch, batch_idx):
        inputs = AttrDict(map_dict(lambda x: x, batch))
        output = self(inputs)
        losses = self.loss(output, inputs)
        if self.tloss is not None:
            self.log_dict(dict(train=self.tloss))
        else:
            self.log_dict(dict(train=losses.total))
        return losses.total

    def validation_step(self, batch, batch_idx):
        val_loss = self.training_step(batch, batch_idx)

        if self.vloss is not None:
            self.log_dict(dict(val=self.vloss))
        else:
            self.log_dict(dict(val=val_loss))

        return val_loss


class DynaTrainer(BaseTrainer):
    def __init__(self, args):
        self.args = args
        self.setup_device()

        # set up params
        self.conf = conf = self.get_config()
        self._hp = self._default_hparams()
        self._hp.overwrite(self.conf.hp)
        self._hp.exp_path = make_path(conf.exp_dir, self.args.path, '', False)
        self.conf = self.postprocess_conf(conf)

        # self.setup_training_monitors()

        vae_conf = self.conf.vae_config
        self.bsize = vae_conf.model_config.batch_size
        self.vae_model = vae_conf.model_class(vae_conf.model_config)
        self.vae_model.to(self.device)
        load_model_weights(self.vae_model, vae_conf.checkpoint)

        self.train_loader = self.get_data_loader('train')
        self.val_loader = self.get_data_loader('val', dataset_size=self.args.val_data_size)

        # create the dynamic model

        sample_data = self.val_loader.dataset[0]
        if self.conf.dyna_conf.cond_skills:
            input_dim = sample_data['states'].shape[-1] + sample_data['skills'].shape[-1]
        else:
            input_dim = sample_data['states'].shape[-1]
        output_dim = sample_data['states'].shape[-1]

        if self._hp.trivial:
            tdset = self.train_loader.dataset.dataset
            vdset = self.val_loader.dataset.dataset
            pred = (tdset['next_states'] - tdset['states']).mean(0)

            tloss = ((pred - (tdset['next_states'] - tdset['states'])) ** 2).sum(-1).mean(0)
            vloss = ((pred - (vdset['next_states'] - vdset['states'])) ** 2).sum(-1).mean(0)
            print(f'[trloss] {tloss}, [vrloss] {vloss}')
            self.dyna_model = DynaModel(input_dim=input_dim, output_dim=output_dim,
                                        **self.conf.dyna_conf, tloss=tloss, vloss=vloss)
            self.dyna_model.to(self.device)
        else:
            self.dyna_model = DynaModel(input_dim=input_dim, output_dim=output_dim,
                                        **self.conf.dyna_conf)
            self.dyna_model.to(self.device)
        self.optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.dyna_model.parameters()), lr=self._hp.lr)

        # load model params from checkpoint
        self.global_step, start_epoch = 0, 0
        # TODO: fix resuming
        # if args.resume or conf.ckpt_path is not None:
        #     start_epoch = self.resume(args.resume, conf.ckpt_path)

        if args.train:
            trainer = pl.Trainer(
                gpus=os.environ.get('CUDA_VISIBLE_DEVICES', None),
                logger=WandbLogger(project='spirl_dyna',
                                   name=self.conf.logger.get('name', None),
                                   save_dir=self._hp.exp_path,
                                   log_model=not self.args.dont_save),
                max_epochs=self._hp.num_epochs,
            )
            trainer.fit(self.dyna_model, self.train_loader, self.val_loader)

        if not self._hp.trivial:
            trloss = self.compute_recon_loss(phase='train')
            vrloss = self.compute_recon_loss(phase='valid')

            print(f'train_reconstruction_loss = {trloss}')
            print(f'valid_reconstruction_loss = {vrloss}')

    def compute_recon_loss(self, phase):
        dloader = self.train_loader if phase == 'train' else self.val_loader

        loss_meter = AverageMeter()
        for batch in dloader:
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), batch))
            # vae model uses states internally to represent states
            inputs.states = inputs.states_traj
            recon = self.vae_model.decode(
                inputs.skills,
                self.vae_model._learned_prior_input(inputs),
                steps=self.vae_model._hp.n_rollout_steps,
                inputs=inputs
            )

            # avg loss across all dims per step and per sample
            recon_loss = ((inputs.actions_traj - recon) ** 2).sum(-1).mean()
            loss_meter.update(recon_loss.item())

        return loss_meter.avg

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def get_config(self):
        conf = AttrDict()

        # paths
        conf.exp_dir = get_exp_dir()
        conf.conf_path = get_config_path(self.args.path)

        # general and agent configs
        print('loading from the config file {}'.format(conf.conf_path))
        conf_module = imp.load_source('conf', conf.conf_path)
        conf.vae_config = conf_module.vae_config
        conf.hp = conf_module.configuration

        conf.dyna_conf = conf_module.dyna_conf

        conf.logger = conf_module.logger_conf

        # data config
        conf.data = conf_module.data_config
        return conf

    def get_data_loader(self, phase, dataset_size=-1):
        data_spec = self.conf.data.dataset_spec
        dataset_class = data_spec.dataset_class

        bsize = self.bsize if phase=='train' else dataset_size
        dataset =  dataset_class(self._hp.data_dir, self.conf.data, phase=phase,
                                 shuffle = (phase == 'train'), dataset_size=dataset_size)
        data_loader = dataset.get_data_loader(bsize, n_repeat=1)

        states, next_states, skills = [], [], []
        states_traj, actions_traj = [], []
        for sample_batched in data_loader:
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            if self._hp.random_latent:
                nz = self.conf.vae_config.model_config.nz_vae
                z = torch.randn(bsize, nz).to(inputs.states)
            else:
                # sample q(z | tau)
                z = self.vae_model._run_inference(inputs).sample()

            states.append(inputs.states[:, 0, :].data)
            next_states.append(inputs.states[:, -1, :].data)
            skills.append(z.data)
            states_traj.append(inputs.states)
            actions_traj.append(inputs.actions)

        latent_dataset = AttrDict(
            states=torch.cat(states, 0),
            next_states=torch.cat(next_states, 0),
            skills=torch.cat(skills, 0),
            states_traj=torch.cat(states_traj, 0),
            actions_traj=torch.cat(actions_traj, 0),
        )

        dset = DynaDataset(latent_dataset)
        dloader = DataLoader(dset, batch_size=bsize, shuffle=(phase=='train'))
        return dloader

    def postprocess_conf(self, conf):
        model_conf = conf.vae_config.model_config
        model_conf['batch_size'] = self._hp.batch_size if not torch.cuda.is_available() \
            else int(self._hp.batch_size / torch.cuda.device_count())
        model_conf.update(conf.data.dataset_spec)
        model_conf['device'] = conf.data['device'] = self.device.type
        conf.dyna_conf.lr = self._hp.lr

        return conf

    def _default_hparams(self):
        # TODO: remove irrelevant parameters
        default_dict = ParamDict({
            'model': None,
            'model_test': None,
            'logger': None,
            'logger_test': None,
            'evaluator': None,
            'data_dir': None,  # directory where dataset is in
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 200,
            'epoch_cycles_train': 1,
            'optimizer': 'radam',    # supported: 'adam', 'radam', 'rmsprop', 'sgd'
            'lr': 1e-3,
            'gradient_clip': None,
            'momentum': 0,      # momentum in RMSProp / SGD optimizer
            'adam_beta': 0.9,       # beta1 param in Adam
            'top_of_n_eval': 1,     # number of samples used at eval time
            'top_comp_metric': None,    # metric that is used for comparison at eval time (e.g. 'mse')
            'random_latent': False,
            'trivial': False
            })
        return default_dict


    def resume(self, ckpt, path=None):
        path = os.path.join(self._hp.exp_path, 'weights') if path is None else os.path.join(path, 'weights')
        assert ckpt is not None  # need to specify resume epoch for loading checkpoint
        weights_file = CheckpointHandler.get_resume_ckpt_file(ckpt, path)
        self.global_step, start_epoch, _ = \
            CheckpointHandler.load_weights(weights_file, self.dyna_model,
                                           load_step=True, load_opt=True, optimizer=self.optimizer,
                                           strict=self.args.strict_weight_loading)
        self.dyna_model.to(self.dyna_model.device)
        return start_epoch

    def get_optimizer_class(self):
        optim = self._hp.optimizer
        if optim == 'adam':
            get_optim = partial(get_clipped_optimizer, optimizer_type=Adam, betas=(self._hp.adam_beta, 0.999))
        elif optim == 'radam':
            get_optim = partial(get_clipped_optimizer, optimizer_type=RAdam, betas=(self._hp.adam_beta, 0.999))
        elif optim == 'rmsprop':
            get_optim = partial(get_clipped_optimizer, optimizer_type=RMSprop, momentum=self._hp.momentum)
        elif optim == 'sgd':
            get_optim = partial(get_clipped_optimizer, optimizer_type=SGD, momentum=self._hp.momentum)
        else:
            raise ValueError("Optimizer '{}' not supported!".format(optim))
        return partial(get_optim, gradient_clip=self._hp.gradient_clip)

    # def setup_training_monitors(self):
    #     self.training_context = autograd.detect_anomaly if self.args.detect_anomaly else dummy_context
    #     self.hooks = []
    #     self.hooks.append(LossSpikeHook('sg_img_mse_train'))
    #     self.hooks.append(NanGradHook(self))
    #     self.hooks.append(NoneGradHook(self))


def load_model_weights(model, checkpoint, epoch='latest'):
    """Loads weights for a given model from the given checkpoint directory."""
    checkpoint_dir = checkpoint if os.path.basename(checkpoint) == 'weights' \
        else os.path.join(checkpoint, 'weights')     # checkpts in 'weights' dir
    checkpoint_path = CheckpointHandler.get_resume_ckpt_file(epoch, checkpoint_dir)
    CheckpointHandler.load_weights(checkpoint_path, model=model)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the config file directory")
    parser.add_argument('--dont_save', default=False, type=int,
                        help="if True, nothing is saved to disk. Note: this doesn't work")  # TODO this doesn't work

    parser.add_argument('--val_data_size', default=-1, type=int,
                        help='number of sequences in the validation set. If -1, the full dataset is used')
    # Running protocol
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--train', default=True, type=int,
                        help='if False, will run one validation epoch')
    parser.add_argument('--skip_first_val', default=False, type=int,
                        help='if True, will skip the first validation epoch')

    # Misc
    parser.add_argument('--gpu', default=-1, type=int,
                        help='will set CUDA_VISIBLE_DEVICES to selected value')
    parser.add_argument('--val_interval', default=5, type=int,
                        help='number of epochs per validation')
    parser.add_argument('--log_interval', default=500, type=int,
                        help='number of updates per training log')

    # Debug
    parser.add_argument('--detect_anomaly', default=False, type=int,
                        help='if True, uses autograd.detect_anomaly()')

    return parser.parse_args()

if __name__ == '__main__':
    DynaTrainer(_parse_args())