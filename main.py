import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy

from runners.rs256_guided_diffusion import Diffusion

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--repeat_run', type=int, default=1, help='Repeat run')
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--r', dest='reverse_steps', type=int, default=20, help='Revserse steps')
    parser.add_argument('--comment', type=str, default='', help='Comment')
    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    os.makedirs(config.log_dir, exist_ok=True)
    if config.model.type == 'conditional':

        dir_name = 'recons_{}_t{}_r{}_w{}'.format(config.data.data_kw,
                                                    args.t, args.reverse_steps,
                                                    config.sampling.guidance_weight)
    else:

        dir_name = 'recons_{}_t{}_r{}_lam{}'.format(config.data.data_kw,
                                                    args.t, args.reverse_steps,
                                                    config.sampling.lambda_)

    if config.model.type == 'conditional':
        print('Use residual gradient guidance during sampling')
        dir_name = 'guided_' + dir_name
    elif config.sampling.lambda_ > 0:
        print('Use residual gradient penalty during sampling')
        dir_name = 'pi_' + dir_name
    else:
        print('Not use physical gradient during sampling')

    log_dir = os.path.join(config.log_dir, dir_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.yml'), 'w') as outfile:
        yaml.dump(config, outfile)

    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, 'logging_info'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, config, logger, log_dir


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config, logger, log_dir = parse_args_and_config()
    print(">" * 80)
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print("<" * 80)

    try:
        runner = Diffusion(args, config, logger, log_dir)
        runner.reconstruct()

    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
