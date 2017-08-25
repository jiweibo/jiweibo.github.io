---
layout: post
title:  "build dataset loader"
date:   2017-08-25 15:55:00 +0800
categories: dataset
location: Harbin, China
description: How to build datasets and dataloader. Tutorial from torch.utils.data
---
---
Build dataset and dataloader

# 概述

在训练机器学习模型中，一般会应用mini-batch的方法来遍历数据集，此外还需要对加载的数据集进行归一化等附加操作。如果能够构建可迭代的数据集模型，并且添加对数据集的附加操作处理接口，那么数据集的使用将非常方便。例如Pytorch提供的对数据集的非常优雅的处理方式：定义附加操作集合，加载数据集。这样即可非常优雅的遍历整个数据集。

{% highlight ruby %}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=True, download=False, transform=transform),
    batch_size=batch_size, shuffle=True)

....
for x_, y_ in train_loader:
    # do something
....
{% endhighlight %}

接下来是对上述代码的源码学习，我们分析Pytorch提供的对数据集处理的源代码，来进一步的理解这种方式。

# 构建生成式数据集

## Dataset

抽象类Dataset，子类要实现方法：``__len__``, ``__getitem__``，使得Dataset对象支持len()和切片操作。

{% highlight ruby %}
class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
{% endhighlight %}

pytorch此模块官方代码见[<u>link</u>](http://pytorch.org/docs/master/_modules/torch/utils/data/dataset.html)


## Sampler

要能以iter的形式获取dataset中的数据，需要采样器sampler，这里介绍SequentialSampler和RandomSampler。  
首先来看Sampler，实现``__iter__``和``__len__``使得支持iter()和len()函数。

{% highlight ruby %}
class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
{% endhighlight %}

SequentialSampler为顺序采样，按照原顺序遍历数据集，采样返回数据集的下标（若数据集长度为3，则依次迭代返回0，1，2）。

{% highlight ruby %}
class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples
{% endhighlight %}

RandomSampler为随机采样，采样返回数据集的下标的随机排序（若数据集长度为3，则迭代返回可能为[（0，1，2），（0，2，1），（1，0，2），（1，2，0），（2，0，1），（2，1，0）]。

{% highlight ruby %}
class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(torch.randperm(self.num_samples).long())
        # return iter(np.random.permutation(self.num_samples))

    def __len__(self):
        return self.num_samples
{% endhighlight %}

源代码中还有SubsetRandomSampler和WeightedRandomSampler，有兴趣的同学点这里[<u>link</u>](http://pytorch.org/docs/master/_modules/torch/utils/data/sampler.html)


## DataLoader && DataLoaderIter

将dataset和sampler组合在一起，构成dataloader，为让其支持迭代操作，还需定义DataLoaderIter（在此省略多进程的部分）。

### DataLoader
{% highlight ruby %}
class DataLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, the ``shuffle`` argument is ignored.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        collate_fn (callable, optional)
        pin_memory (bool, optional)
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If False and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0,
                 collate_fn=default_collate, pin_memory=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset)
        elif not shuffle:
            self.sampler = SequentialSampler(dataset)

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) /// self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) /// self.batch_size

{% endhighlight %}

### DataLoaderIter
{% highlight ruby %}
class DataLoaderIter(object):
    """Iterates once over the DataLoader's dataset, as specified by the sampler"""

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.sampler = loader.sampler
        self.drop_last = loader.drop_last

        self.samples_remaining = len(self.sampler)
        self.sample_iter = iter(self.sampler)

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) /// self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) /// self.batch_size

    def __next__(self):
        if self.drop_last and self.samples_remaining < self.batch_size:
            raise StopIteration
        if self.samples_remaining == 0:
            raise StopIteration
        indices = self._next_indices()
        batch = [self.dataset[i] for i in indices]
        return batch

    next = __next__()

    def __iter__(self):
        return self

    def _next_indices(self):
        batch_size = min(self.samples_remaining, self.batch_size)
        batch = [next(self.sample_iter) for _ in range(batch_size)]
        self.samples_remaining -= len(batch)
        return batch
{% endhighlight %}

以上的处理完成了可迭代的生成式数据集的构建完整代码见[<u>link</u>](http://pytorch.org/docs/master/_modules/torch/utils/data/dataloader.html#DataLoader)，接下来构建mnist的dataset。


# MNIST Dateset

MNIST数据集继承上述介绍的Dataset类，需要实现``__len__``和``__getitem__``方法：这里的``__len__``方法取决于MNIST数据集，60000的训练集和10000的测试集；``getitem``方法使得数据集支持切片操作，并且在此可完成transform等附加操作。此处完整代码见torchvision提供的源码。

{% highlight ruby %}
def __len__(self):
    if self.train:
        return 60000
    else:
        return 10000

def __getitem__(self, index):
    if self.train:
        img, target = self.train_data[index], self.train_labels[index]
    else:
        img, target = self.test_data[index], self.test_labels[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img.numpy(), mode='L')

    if self.transform is not None:
        img = self.transform(img)

    if self.target_transform is not None:
        target = self.target_transform(target)

    return img, target
{% endhighlight %}



# Transform

transfrom模块中定义了许多预处理操作，这里介绍几个简单常用的。

## Compose
相当于是一个容器，顺序执行容器内部的操作。

{% highlight ruby %}
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
{% endhighlight %}

## ToTenser
Pytorch最常用的方法，将numpy.ndarray （H x W x C）[0, 255] 转成tensor (C x H x W) [0.0, 1.0]。

{% highlight ruby %}
class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backard compability
            return img.float().div(255)
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img
{% endhighlight %}


## Normalize

Normalize是归一化常用的操作，具体见代码``__doc__``

{% highlight ruby %}
class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor
{% endhighlight %}


# 总结

现在我们再来看这一段代码，是不是感觉特别的清晰：定义预处理操作ToTensor、Normalize，将数据限制到[-1, 1]范围；将mnist的dataset传递给dataloader构造loader生成模式，这样就能够在训练代码中使用for··in··语句直接加载数据。

{% highlight ruby %}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=True, download=False, transform=transform),
    batch_size=batch_size, shuffle=True)

....
for x_, y_ in train_loader:
    # do something
....
{% endhighlight %}
