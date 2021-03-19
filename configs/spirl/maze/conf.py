import os

from spirl.models.skill_prior_mdl import SkillPriorMdl, SkillSpaceLogger
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.maze import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': SkillPriorMdl,
    'logger': SkillSpaceLogger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'maze'),
    'epoch_cycles_train': 1,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 1,
    'top_comp_metric': 'mse',
    'num_epochs': 200,
    'batch_size': 1024,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=1e-2,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1
