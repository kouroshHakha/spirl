import os
import copy

from spirl.utils.general_utils import AttrDict
from spirl.rl.components.agent import FixedIntervalHierarchicalAgent
from spirl.rl.policies.mlp_policies import MLPPolicy
from spirl.rl.components.critic import MLPCritic
from spirl.rl.envs.kitchen import KitchenEnv
from spirl.rl.components.sampler import HierarchicalSampler
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.skill_space_agent import SkillSpaceAgent
from spirl.configs.default_data_configs.kitchen import data_spec
from spirl.models.skill_prior_mdl import SkillPriorMdl



current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'Testing dynamic prediction with latent on kitchen env'


configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': KitchenEnv,
    'sampler': HierarchicalSampler,
    'data_dir': '.',
    'num_epochs': 100,
    'max_rollout_len': 280,
    'n_steps_per_epoch': 100000,
    'n_warmup_steps': 5e3,
    'batch_size': 256
}
configuration = AttrDict(configuration)

# Replay Buffer
replay_params = AttrDict(
)

# Observation Normalization
obs_norm_params = AttrDict(
)

logger_conf = AttrDict(name='noSkill')

base_agent_params = AttrDict(
    batch_size=configuration.batch_size,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    # obs_normalizer=Normalizer,
    # obs_normalizer_params=obs_norm_params,
    clip_q_target=False,
)

model_params = AttrDict(
        batch_size=configuration.batch_size,
        state_dim=data_spec.state_dim,
        action_dim=data_spec.n_actions,
        kl_div_weight=5e-4,
        nz_enc=128,
        nz_mid=128,
        n_processing_layers=5,
        nz_vae=10,
        n_rollout_steps=10,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_params.n_rollout_steps + 1  # flat last action from seq gets cropped

vae_config = AttrDict(
    model_class=SkillPriorMdl,
    model_config=model_params,
    checkpoint=os.path.join(os.environ["EXP_DIR"], "kitchen_state/spirl")
)

dyna_conf = AttrDict(
    hidden_dim=200,
    hidden_depth=3,
    output_mod=None,
    cond_skills=False
)
