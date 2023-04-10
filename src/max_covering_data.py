import torch
from pathlib import Path
import re
import urllib.request
import random


def get_random_dataset(num_items, num_sets, seed):
    random.seed(seed)
    dataset = []
    for i in range(100):
        weights = [random.randint(1, 100) for _ in range(num_items)]
        sets = []
        for set_idx in range(num_sets):
            covered_items = random.randint(10, 30)
            sets.append(random.sample(range(num_items), covered_items))
        dataset.append((f'rand{i}', weights, sets))
    return dataset


def get_twitch_dataset():
    import math
    dataset = []
    languages = ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU']
    for language in languages:
        with open(f'data/twitch/{language}/musae_{language}_edges.csv') as f:
            edges = []
            node_ids = set()
            for e in f.readlines():
                e_str = e.strip().split(',')
                if e_str[0] == 'from' and e_str[1] == 'to':
                    continue
                n1, n2 = int(e_str[0]), int(e_str[1])
                edges.append((n1, n2))
                node_ids.add(n1)
                node_ids.add(n2)
        id_map = {n: i for i, n in enumerate(node_ids)}
        weights = [-1 for _ in node_ids]
        with open(f'data/twitch/{language}/musae_{language}_target.csv') as f:
            for line in f.readlines():
                line_str = line.strip().split(',')
                if line_str[0] == 'id':
                    continue
                weights[id_map[int(line_str[5])]] = math.floor(math.log(int(line_str[3]) + 1))
        assert min(weights) >= 0
        sets = [[] for _ in node_ids]
        for n1, n2 in edges:
            sets[id_map[n1]].append(id_map[n2])
        dataset.append((language, weights, sets))
    return dataset


ONLINE_REPO = 'http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/'


problem_set = {
    'scp4': 10,
    'scp5': 10,
    'scp6': 5,
    'scpa': 5,
    'scpb': 5,
    'scpc': 5,
    'scpd': 5,
    'scpe': 5,
    'scpnre': 5,
    'scpnrf': 5,
    'scpnrg': 5,
    'scpnrh': 5
}


class SCP_ORLIB:
    def __init__(self, fetch_online=False):
        super(SCP_ORLIB, self).__init__()
        self.classes = problem_set.keys()
        self.data_list = []
        self.data_path = Path('../data/scp_orlib')

        for cls in self.classes:
            cls_len = problem_set[cls]
            for i in range(cls_len):
                self.data_list.append(cls + '{}'.format(i + 1))

        # define compare function
        def name_cmp(a, b):
            a = re.findall(r'[0-9]+|[a-z]+', a)
            b = re.findall(r'[0-9]+|[a-z]+', b)
            for _a, _b in zip(a, b):
                if _a.isdigit() and _b.isdigit():
                    _a = int(_a)
                    _b = int(_b)
                cmp = (_a > _b) - (_a < _b)
                if cmp != 0:
                    return cmp
            if len(a) > len(b):
                return -1
            elif len(a) < len(b):
                return 1
            else:
                return 0

        def cmp_to_key(mycmp):
            'Convert a cmp= function into a key= function'
            class K:
                def __init__(self, obj, *args):
                    self.obj = obj
                def __lt__(self, other):
                    return mycmp(self.obj, other.obj) < 0
                def __gt__(self, other):
                    return mycmp(self.obj, other.obj) > 0
                def __eq__(self, other):
                    return mycmp(self.obj, other.obj) == 0
                def __le__(self, other):
                    return mycmp(self.obj, other.obj) <= 0
                def __ge__(self, other):
                    return mycmp(self.obj, other.obj) >= 0
                def __ne__(self, other):
                    return mycmp(self.obj, other.obj) != 0
            return K

        # sort data list according to the names
        self.data_list.sort(key=cmp_to_key(name_cmp))
        print(self.data_list)

        fetched_flag = self.data_path / 'fetched_online'

        if fetch_online or not fetched_flag.exists():
            self.__fetch_online()
            fetched_flag.touch()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """Notice: the indices start from 0 which is different from the original ORLIB format (start from 1)"""
        name = self.data_list[idx]

        dat_path = self.data_path / (name + '.txt')
        dat_file = dat_path.open()

        def split_line(x):
            for _ in re.split(r'[,\s]', x.rstrip('\n')):
                if _ == "":
                    continue
                else:
                    yield int(_)

        dat_list = [[_ for _ in split_line(line)] for line in dat_file]

        nrows, ncols = dat_list[0]

        # read data
        row_idx = 1

        # read column weights
        column_weights = []
        while len(column_weights) < ncols:
            column_weights += dat_list[row_idx]
            row_idx += 1

        assert len(column_weights) == ncols

        # read row sets
        row_sets = []
        remain_len_of_this_row = 0
        while row_idx < len(dat_list):
            if remain_len_of_this_row == 0:
                assert len(dat_list[row_idx]) == 1
                remain_len_of_this_row = dat_list[row_idx][0]
                row_sets.append([])
            else:
                row_sets[-1] += [item-1 for item in dat_list[row_idx]]  # we let the index start from 0
                remain_len_of_this_row -= len(dat_list[row_idx])
                assert remain_len_of_this_row >= 0
            row_idx += 1

        return name, column_weights, row_sets

    def __fetch_online(self):
        """
        Fetch from online QAPLIB data
        """
        for name in self.data_list:
            dat_content = urllib.request.urlopen(ONLINE_REPO + '{}.txt'.format(name)).read()

            dat_file = (self.data_path / (name + '.txt')).open('wb')
            dat_file.write(dat_content)
