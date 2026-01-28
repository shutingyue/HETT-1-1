import math
import sys
import time

from PIL import Image


# from gsamllavanav.parser import ExperimentArgs


_active = False
_silent = True

def write_to_record_file(data, file_path, verbose=True):
    if verbose:
        print(data)
    record_file = open(file_path, 'a')
    record_file.write(data+'\n')
    record_file.close()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

class Timer:
    def __init__(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def reset(self):
        self.cul = OrderedDict()
        self.start = {}
        self.iter = 0

    def tic(self, key):
        self.start[key] = time.time()

    def toc(self, key):
        delta = time.time() - self.start[key]
        if key not in self.cul:
            self.cul[key] = delta
        else:
            self.cul[key] += delta

    def step(self):
        self.iter += 1

    def show(self):
        total = sum(self.cul.values())
        for key in self.cul:
            print("%s, total time %0.2f, avg time %0.2f, part of %0.2f" %
                  (key, self.cul[key], self.cul[key]*1./self.iter, self.cul[key]*1./total))
        print(total / self.iter)

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


# def init(args: ExperimentArgs):
#     global _active
#     global _silent
#
#     _active = args.log
#     _silent = args.silent
#
#     if _active:
#         if args.resume_log_id:
#             wandb.init(entity='water-cookie', project='citynav', id=args.resume_log_id, resume="must")
#         else:
#             wandb.init(project='citynav', config=args.to_dict())
#
#
# def define_metric(name: str, step_metric: str = None, summary: str = None):
#     if _active:
#         wandb.define_metric(name, step_metric, summary=summary)
#
#
# def log(data, step=None, commit=None):
#     if _active:
#         wandb.log(data, step=step, commit=commit)
#     if not _silent:
#         print(data)
#
#
# def log_images(name: str, images: list[Image.Image], captions: list[str], max_n=10, step=None, commit=None):
#     if _active:
#         images = [wandb.Image(image, caption=caption) for image, caption in zip(images[:max_n], captions[:max_n])]
#         wandb.log({name: images}, step=step, commit=commit)
#
#
# def finish():
#     if _active:
#         wandb.finish()
