import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
import copy
from train import rollout
from utils import get_inner_model


class Baseline(object):
    """
    Abstract base class for baseline methods in reinforcement learning.
    Baselines are used to reduce variance in policy gradient methods by providing
    a reference point for advantage estimation.
    """

    def wrap_dataset(self, dataset):
        """
        Wrap the dataset to include baseline information.
        
        Args:
            dataset: Original dataset
            
        Returns:
            Wrapped dataset with baseline information
        """
        return dataset

    def unwrap_batch(self, batch):
        """
        Extract data and baseline from a wrapped batch.
        
        Args:
            batch: Wrapped batch containing data and baseline
            
        Returns:
            Tuple of (data, baseline)
        """
        return batch, None

    def eval(self, x, c):
        """
        Evaluate the baseline for given inputs and costs.
        
        Args:
            x: Input data
            c: Cost/reward values
            
        Returns:
            Baseline value and loss (if any)
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        """
        Get learnable parameters for the baseline.
        
        Returns:
            List of learnable parameters
        """
        return []

    def epoch_callback(self, model, epoch):
        """
        Callback function called at the end of each epoch.
        
        Args:
            model: Current model
            epoch: Current epoch number
        """
        pass

    def state_dict(self):
        """
        Get the state dictionary for checkpointing.
        
        Returns:
            State dictionary
        """
        return {}

    def load_state_dict(self, state_dict):
        """
        Load state from checkpoint.
        
        Args:
            state_dict: State dictionary to load
        """
        pass


class WarmupBaseline(Baseline):
    """
    Baseline that gradually transitions from a warmup baseline to the main baseline.
    This helps stabilize training in the early epochs.
    """

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8):
        """
        Initialize warmup baseline.
        
        Args:
            baseline: Main baseline to transition to
            n_epochs: Number of epochs for warmup
            warmup_exp_beta: Beta parameter for exponential baseline during warmup
        """
        super(Baseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        """
        Wrap dataset using appropriate baseline based on warmup stage.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Wrapped dataset
        """
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        """
        Unwrap batch using appropriate baseline based on warmup stage.
        
        Args:
            batch: Wrapped batch
            
        Returns:
            Unwrapped data and baseline
        """
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):
        """
        Evaluate baseline using convex combination during warmup.
        
        Args:
            x: Input data
            c: Cost/reward values
            
        Returns:
            Convex combination of main baseline and warmup baseline
        """
        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, l = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha) * lw

    def epoch_callback(self, model, epoch):
        """
        Update warmup alpha and call inner baseline callback.
        
        Args:
            model: Current model
            epoch: Current epoch number
        """
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        self.alpha = (epoch + 1) / float(self.n_epochs)
        if epoch < self.n_epochs:
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        """
        Save only the inner baseline state.
        
        Returns:
            State dictionary of inner baseline
        """
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load only the inner baseline state.
        
        Args:
            state_dict: State dictionary to load
        """
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class NoBaseline(Baseline):
    """
    Baseline that provides no baseline value (always returns 0).
    This is equivalent to not using a baseline at all.
    """

    def eval(self, x, c):
        """
        Return zero baseline and zero loss.
        
        Args:
            x: Input data (unused)
            c: Cost/reward values (unused)
            
        Returns:
            Tuple of (0, 0) - no baseline, no loss
        """
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):
    """
    Baseline that maintains an exponentially moving average of costs.
    This provides a simple but effective baseline for many problems.
    """

    def __init__(self, beta):
        """
        Initialize exponential baseline.
        
        Args:
            beta: Exponential moving average parameter (0 < beta < 1)
        """
        super(Baseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, c):
        """
        Evaluate baseline using exponential moving average.
        
        Args:
            x: Input data (unused)
            c: Cost/reward values
            
        Returns:
            Tuple of (baseline value, 0) - no loss
        """
        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        """
        Save the current baseline value.
        
        Returns:
            State dictionary containing baseline value
        """
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        """
        Load the baseline value from checkpoint.
        
        Args:
            state_dict: State dictionary to load
        """
        self.v = state_dict['v']


class CriticBaseline(Baseline):
    """
    Baseline that uses a learned critic network to estimate state values.
    This provides a more sophisticated baseline that can generalize better.
    """

    def __init__(self, critic):
        """
        Initialize critic baseline.
        
        Args:
            critic: Neural network that estimates state values
        """
        super(Baseline, self).__init__()

        self.critic = critic

    def eval(self, x, c):
        """
        Evaluate baseline using critic network.
        
        Args:
            x: Input data
            c: Cost/reward values
            
        Returns:
            Tuple of (baseline value, MSE loss for critic training)
        """
        v = self.critic(x)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v, c.detach())

    def get_learnable_parameters(self):
        """
        Get learnable parameters from critic network.
        
        Returns:
            List of critic network parameters
        """
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        """
        No epoch callback needed for critic baseline.
        
        Args:
            model: Current model (unused)
            epoch: Current epoch (unused)
        """
        pass

    def state_dict(self):
        """
        Save critic network state.
        
        Returns:
            State dictionary containing critic state
        """
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        """
        Load critic network state with backward compatibility.
        
        Args:
            state_dict: State dictionary to load
        """
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


class RolloutBaseline(Baseline):
    """
    Baseline that uses a separate model to generate baseline values through rollouts.
    This baseline can be updated when the main model improves, providing a moving target.
    """

    def __init__(self, model, problem, opts, epoch=0):
        """
        Initialize rollout baseline.
        
        Args:
            model: Model to use for generating baseline values
            problem: Problem instance
            opts: Options/configuration
            epoch: Starting epoch number
        """
        super(Baseline, self).__init__()

        self.problem = problem
        self.opts = opts

        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        """
        Update the baseline model and regenerate baseline values.
        
        Args:
            model: New model to use as baseline
            epoch: Current epoch number
            dataset: Optional dataset to use (if None, generate new one)
        """
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset

        if dataset is not None:
            if len(dataset) != self.opts.val_size:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
            elif (dataset[0]["users"]).size(0) != self.opts.n_users:
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None

        if dataset is None:
            self.dataset = self.problem.make_dataset(
                n_users=self.opts.n_users, n_facilities=self.opts.n_facilities, num_samples=self.opts.val_size,
                filename='data/MCLP_1000_30_normal_Normalization.pkl',
                p=self.opts.p, r=self.opts.r, distribution=self.opts.data_distribution)
            print(f"p = {self.opts.p}")
        else:
            self.dataset = dataset
        print("Evaluating baseline model on evaluation dataset")
        self.bl_vals = rollout(self.model, self.dataset, self.opts).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch
        print("_update_model executed, epoch {}, mean {}".format(self.epoch, self.mean))

    def wrap_dataset(self, dataset):
        """
        Wrap dataset with baseline values computed by the baseline model.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset wrapped with baseline values
        """
        print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        return BaselineDataset(dataset, rollout(self.model, dataset, self.opts).view(-1, 1))

    def unwrap_batch(self, batch):
        """
        Extract data and baseline from wrapped batch.
        
        Args:
            batch: Wrapped batch containing data and baseline
            
        Returns:
            Tuple of (data, flattened baseline)
        """
        return batch['data'], batch['baseline'].view(-1)  # Flatten result to undo wrapping as 2D

    def eval(self, x, c):
        """
        Evaluate baseline using the baseline model.
        
        Args:
            x: Input data
            c: Cost/reward values (unused)
            
        Returns:
            Tuple of (baseline value, 0) - no loss
        """
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.model(x)

        # There is no loss
        return v, 0

    def epoch_callback(self, model, epoch):
        """
        Challenge the current baseline with the model and replace if improved.
        Uses statistical significance testing to determine if update is warranted.
        
        Args:
            model: The model to challenge the baseline
            epoch: The current epoch
        """
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(model, self.dataset, self.opts).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        if candidate_mean - self.mean < 0:
            # Calc p value
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        """
        Save baseline model, dataset, and epoch information.
        
        Returns:
            State dictionary containing baseline components
        """
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        """
        Load baseline state with DataParallel compatibility.
        
        Args:
            state_dict: State dictionary to load
        """
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])


class BaselineDataset(Dataset):
    """
    Dataset wrapper that includes baseline values alongside original data.
    This allows efficient batching of data and baseline pairs.
    """

    def __init__(self, dataset=None, baseline=None):
        """
        Initialize baseline dataset.
        
        Args:
            dataset: Original dataset
            baseline: Baseline values corresponding to dataset items
        """
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.baseline = baseline
        assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        """
        Get item with both data and baseline.
        
        Args:
            item: Index of item to retrieve
            
        Returns:
            Dictionary containing data and baseline
        """
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }

    def __len__(self):
        """
        Get length of dataset.
        
        Returns:
            Number of items in dataset
        """
        return len(self.dataset)
