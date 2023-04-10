from easydict import EasyDict as edict
import yaml
import argparse


def load_config():
    parser = argparse.ArgumentParser(description='CardNN experiment protocol.')
    parser.add_argument('--cfg', '--config', dest='cfg_file',
                        help='path to the configuration file', default=None, type=str)
    args = parser.parse_args()
    if args.cfg_file is None:
        raise ValueError('Please specify path to the configuration file!')
    return cfg_from_file(args.cfg_file)


def cfg_from_file(filename, cfg=None):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.full_load(f))

    if cfg is None:
        cfg = edict()

    _merge_a_into_b(yaml_cfg, cfg)
    return cfg


def _merge_a_into_b(a, b, strict=False):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        if strict:
            # a must specify keys that are in b
            if k not in b:
                raise KeyError('{} is not a valid config key'.format(k))

            # the types must match, too
            if type(b[k]) is not type(v):
                if type(b[k]) is float and type(v) is int:
                    v = float(v)
                else:
                    if not k in ['CLASS']:
                        raise ValueError('Type mismatch ({} vs. {}) for config key: {}'.format(type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v
