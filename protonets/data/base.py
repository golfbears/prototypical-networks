import torch

def convert_dict(k, v):
    return { k: v }

class CudaTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        for k,v in data.items():
            if hasattr(v, 'cuda'):
                data[k] = v.cuda()

        return data

class SequentialBatchSampler(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __len__(self):
        return self.n_classes

    def __iter__(self):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


def extract_episode(n_support, n_query, d):
    # data: N x C x H x W
    n_examples = d['data'].size(0)

    if n_query == -1:
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }