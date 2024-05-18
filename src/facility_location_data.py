import torch


def get_random_data(num_data, dim, seed, device, demand=False, max_demand=0.5):
    torch.random.manual_seed(seed)
    dataset = []
    for _ in range(100):
        if not demand:
            item = (f'rand{_}', torch.rand(num_data, dim, device=device), None)
        else:
            item = (f'rand{_}', torch.rand(num_data, dim, device=device), torch.ones((num_data,), device=device) * max_demand)
        dataset.append(item)
    return dataset


def get_starbucks_data(device):
    dataset = []
    areas = ['london', 'newyork', 'shanghai', 'seoul']
    for area in areas:
        with open(f'data/starbucks/{area}.csv', encoding='utf-8-sig') as f:
            locations = []
            for l in f.readlines():
                l_str = l.strip().split(',')
                if l_str[0] == 'latitude' and l_str[1] == 'longitude':
                    continue
                n1, n2 = float(l_str[0]) / 365 * 400, float(l_str[1]) / 365 * 400  # real-world coordinates: x100km
                locations.append((n1, n2))
        dataset.append((area, torch.tensor(locations, device=device), None))
    return dataset
