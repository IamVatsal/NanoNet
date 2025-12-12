import numpy as np
from enum import Enum

class _DecayType(Enum):
    # TIME_BASED = "time_based"

    # EPoch-based decay types
    EXPONENTIAL_DECAY = "exponential_decay"
    STEP_DECAY = "step_decay"

    # iteration-based decay types
    CYCLIC_DECAY = "cyclic_decay"

class LRScheduler:
    def __init__(self, base_lr=0.01, decay_type=None, **kwargs):
        self.__base_lr = base_lr
        self.__current_lr = base_lr
        self.__available_decay_types = [decay.value for decay in _DecayType]
        self.__current_decay_type = None

        # Initialize parameters for Tracking epoch and iteration
        self.__current_epoch = 0
        self.__current_iteration = 0

        if decay_type is None:
            self.__current_decay_type = None
        elif decay_type == "exponential_decay":
            self.__current_decay_type = _DecayType.EXPONENTIAL_DECAY
            self.__decay_rate = kwargs.get("decay_rate", 0.1)
        elif decay_type == "step_decay":
            self.__current_decay_type = _DecayType.STEP_DECAY
            self.__drop = kwargs.get("drop", 0.5)
            self.__step_size = kwargs.get("step_size", 10)
        elif decay_type == "cyclic_decay":
            self.__current_decay_type = _DecayType.CYCLIC_DECAY
            self.__min_lr = kwargs.get("min_lr", 0.001)
            self.__max_lr = kwargs.get("max_lr", 0.006)
            self.__cycle_steps = kwargs.get("cycle_steps", 2000)
        else:
            raise ValueError("Unsupported decay type")
        
    def __exponentially_decay_lr(self, base_lr, decay_rate, epoch):
        return base_lr * np.exp(-decay_rate * epoch)
    
    def __step_decay_lr(self, base_lr, epoch, drop=0.5, step_size=10):
        return base_lr * (drop ** np.floor(epoch / step_size))
    
    def __cyclic_decay_lr(self, iteration, min_lr, max_lr, cycle_steps):

        if cycle_steps <= 0:
            raise ValueError("cycle_steps must be a positive integer")
        elif min_lr >= max_lr:
            raise ValueError("min_lr must be less than max_lr")
        elif iteration is None:
            raise ValueError("iteration must be provided for cyclic decay")
        elif iteration < 0:
            raise ValueError("iteration must be a non-negative integer")
        
        cycle = np.floor(1 + iteration / (2 * cycle_steps))
        x = np.abs(iteration / cycle_steps - 2 * cycle + 1)
        lr = min_lr + (max_lr - min_lr) * max(0, (1 - x))
        return lr
    
    def update_lr(self, epoch=None, iteration=None):
        if self.__current_decay_type is None:
            self.__current_lr = self.__base_lr
        elif self.__current_decay_type == _DecayType.EXPONENTIAL_DECAY:
            self.__current_lr = self.__exponentially_decay_lr(self.__base_lr, self.__decay_rate, epoch)
        elif self.__current_decay_type == _DecayType.STEP_DECAY:
            self.__current_lr = self.__step_decay_lr(self.__base_lr, epoch, self.__drop, self.__step_size)
        elif self.__current_decay_type == _DecayType.CYCLIC_DECAY:
            self.__current_lr = self.__cyclic_decay_lr(iteration, self.__min_lr, self.__max_lr, self.__cycle_steps)
        else:
            self.__current_lr = self.__base_lr
        
    def get_current_decay_type(self):
        return self.__current_decay_type
    
    def set_decay_type(self, decay_type, **kwargs):
        if decay_type is None:
            self.__current_decay_type = None
        elif decay_type == "exponential_decay":
            self.__current_decay_type = _DecayType.EXPONENTIAL_DECAY
            self.__decay_rate = kwargs.get("decay_rate", 0.1)
        elif decay_type == "step_decay":
            self.__current_decay_type = _DecayType.STEP_DECAY
            self.__drop = kwargs.get("drop", 0.5)
            self.__step_size = kwargs.get("step_size", 10)
        elif decay_type == "cyclic_decay":
            self.__current_decay_type = _DecayType.CYCLIC_DECAY
            self.__min_lr = kwargs.get("min_lr", 0.001)
            self.__max_lr = kwargs.get("max_lr", 0.006)
            self.__cycle_steps = kwargs.get("cycle_steps", 2000)
        else:
            raise ValueError("Unsupported decay type")
        
    def available_decay_types(self):
        print(f"Available Decay Types: {self.__available_decay_types}")
        
    def get_base_lr(self):
        return self.__base_lr
    
    def set_base_lr(self, lr):
        self.__base_lr = lr

    def step(self):
        self.__current_iteration += 1
        if self.__current_decay_type == _DecayType.CYCLIC_DECAY:
            self.update_lr(iteration=self.__current_iteration)

    def epoch_step(self):
        self.__current_epoch += 1
        self.__current_iteration = 0
        if self.__current_decay_type in [_DecayType.EXPONENTIAL_DECAY, _DecayType.STEP_DECAY]:
            self.update_lr(epoch=self.__current_epoch)

    @property
    def lr(self):
        return self.__current_lr

    def get_params(self):
        params = {"base_lr": self.__base_lr}
        if self.__current_decay_type == _DecayType.EXPONENTIAL_DECAY:
            params.update({
                "decay_type": "exponential_decay",
                "decay_rate": self.__decay_rate
            })
        elif self.__current_decay_type == _DecayType.STEP_DECAY:
            params.update({
                "decay_type": "step_decay",
                "drop": self.__drop,
                "step_size": self.__step_size
            })
        elif self.__current_decay_type == _DecayType.CYCLIC_DECAY:
            params.update({
                "decay_type": "cyclic_decay",
                "min_lr": self.__min_lr,
                "max_lr": self.__max_lr,
                "cycle_steps": self.__cycle_steps
            })
        else:
            params["decay_type"] = None
        return params