from PIL import Image
import wandb

from vlnce.parser import ExperimentArgs


_active = False
_silent = True


def init(args: ExperimentArgs):
    global _active
    global _silent
    
    _active = args.log
    _silent = args.silent
    
    if _active:
        if args.resume_log_id:
            wandb.init(entity='water-cookie', project='citynav', id=args.resume_log_id, resume="must")
        else:
            wandb.init(project='citynav', config=args.to_dict())


def define_metric(name: str, step_metric: str = None, summary: str = None):
    if _active:
        wandb.define_metric(name, step_metric, summary=summary)


def log(data, step=None, commit=None):
    if _active:
        wandb.log(data, step=step, commit=commit)
    if not _silent:
        print(data)


def log_images(name: str, images: list[Image.Image], captions: list[str], max_n=10, step=None, commit=None):
    if _active:
        images = [wandb.Image(image, caption=caption) for image, caption in zip(images[:max_n], captions[:max_n])]
        wandb.log({name: images}, step=step, commit=commit)


def finish():
    if _active:
        wandb.finish()


class RunningMeter(object):
    """ running meteor of a scalar value
        (useful for monitoring training loss)
    """
    def __init__(self, name, val=None, smooth=0.99):
        self._name = name
        self._sm = smooth
        self._val = val

    def __call__(self, value):
        val = (value if self._val is None
               else value*(1-self._sm) + self._val*self._sm)
        if not math.isnan(val):
            self._val = val

    def __str__(self):
        return f'{self._name}: {self._val:.4f}'

    @property
    def val(self):
        if self._val is None:
            return 0
        return self._val

    @property
    def name(self):
        return self._name