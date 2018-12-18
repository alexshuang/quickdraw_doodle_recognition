![https://www.kaggle.com/c/quickdraw-doodle-recognition](https://upload-images.jianshu.io/upload_images/13575947-e5b24fdf61d51dcc.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[Quick, Draw!](https://quickdraw.withgoogle.com/)是Google推出的一款很好玩的AI涂鸦游戏，玩家需要在20秒内画出指定内容，例如鸭子、冰箱、苹果等，它的神经网络会实时识别你的涂鸦。

最近Google在kaggle上发布了优化QuickDraw预测识别能力的比赛，[Quick, Draw! Doodle Recognition Challenge](https://www.kaggle.com/c/quickdraw-doodle-recognition)，本篇博文就来是分享涂鸦识别的心得。与以往同类比赛不同的是，这次的数据集规模特别庞大，共有340个分类，总计将近5000万个样本！我在r5 6cores CPU + RTX2070 8G GPU + NVME SSD的机器上训练1 epoch就用时18个小时! 这个比赛的算力门槛很高，如果没有足够的人力和算力支持，那再好的模型也出不来。AI的世界也要拼爹。

为了不至于让你被算力劝退，本文提供的[notebook](https://github.com/alexshuang/quickdraw_doodle_recognition/blob/master/QuickDraw_starter.ipynb)只取一小部分数据参与训练，你可以根据实际情况调整数据量。github: [here](https://github.com/alexshuang/quickdraw_doodle_recognition)

## Read the code / [Notebook](https://github.com/alexshuang/quickdraw_doodle_recognition/blob/master/QuickDraw_starter.ipynb)

```
def get_count(path): return (path.stem, pd.read_csv(path).shape[0])
with ThreadPoolExecutor(2) as e: counts = list(e.map(get_count, TRN_PATH.iterdir()))
counts = sorted(counts, key=lambda x: x[1])

len(counts), counts[0], counts[-1], np.mean([o[1] for o in counts]).astype(np.int)

(340, ('panda', 113613), ('snowman', 340029), 146198)
```

前文说过这个数据集特别庞大，我把每个分类的样本数统计到counts变量，共340个分类，平均每个分类提供了14万6千个训练样本，最少的样本数分类也达到11万，少数几个分类的样本数则超过30万。

![Figure 1: sample counts](https://upload-images.jianshu.io/upload_images/13575947-850e92622a7ac33f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这是样本分类不均的数据集，如果不打算训练完所有的样本，可以每个分类只取最多15万个样本。

## Prepare data

![train.csv & test.csv](https://upload-images.jianshu.io/upload_images/13575947-e03446f43a877c78.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

数据集中重要的字段有countrycode、drawing、recognized，其中最核心字段是drawing，它就是构成图形的所有笔画中点的集合，drawing字段中的数值是这些点的x、y坐标，将同一笔画中所有临近的两个点用直线相连就完成了一笔，所有笔画组合在一起就是一个完整的图形。

```
# https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892
BASE_SIZE = 256
def draw_cv2(raw_strokes, size=128, lw=6, time_color=False):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img

fig, axes = plt.subplots(3, 4, figsize=(8, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(draw_cv2(eval(df.loc[i, 'drawing'])))
plt.tight_layout()
```
![](https://upload-images.jianshu.io/upload_images/13575947-38204ff9e2ce8ccd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

draw_cv2()用于decode drawing字段生成image array。这里我读取的是"panda"分类，并生成头12个样本的图形。可以看到，能看出来带有熊猫特征的只有第三排中间两个样本，换句话说，样本中的噪声信号很大。如果你不打算训练所有的样本，建议只选取recognized == True的样本。recognized == False指的是Quickdraw识别不出来的样本，识别不出来的原因有：一、画错了（噪声） 二、画对了但识别不出来，后者是模型的优化目标。如果想要对recognized == False有更直观的体验，建议你到游戏里玩两把。

解码drawing的方式有两种：
* 训练前预根据drawing生成image文件。
优点：少了中间的转换环节，模型训练速度快。
缺点：需要大量空间：~250G，而且因为总样本数接近5000万，文件系统很可能需要重新格式化，增加inode数最大容量（将blocksize设置为4096），否则很可能出现分区还有空间，但却不能再创建新文件。可以通过命令：$df -i查看文件系统的max inode。
```
def df_to_ims(path):
    df = pd.read_csv(path, nrows=k)
    kids = df.key_id.values
    drawings = df.drawing.values
    words = np.array(['_'.join(o.split()) for o in df.word.values])
    for kid, drawing, word in zip(kids, drawings, words):
        img = draw_cv2(eval(drawing), size=sz)
        img_rgb = np.repeat(img[:, :, None], 3, -1)
        fpth = PATH/f'train_{sz}/{word}/{kid}.png'
        os.makedirs(str(fpth.parent), exist_ok=True)
        plt.imsave(str(fpth), img_rgb)

with ThreadPoolExecutor(12) as e: e.map(df_to_ims, TRN_PATH.iterdir())
```
* 训练前先将*.csv中的drawing取出，为每个样本生成一个单独的drawing.txt文件，训练时动态decode drawing生成image array。
优点：drawing.txt只保存drawing string，存储成本远小于image file（inode限制问题依旧存在）。
缺点：每次读取训练样本都需要decode drawing，需要额外的CPU时间，CPU速度跟不上时会拉慢整个训练进度（建议I7 8 cores）。
```
def draw_cv2(raw_strokes, size=256, lw=7, time_color=True):
    ......

def gen_img(path):
    with open(path) as f: drawing = eval(f.read())
    return draw_cv2(drawing, size=sz)

def open_image(fn):
    ......
    try:
        return gen_img(str(fn)).astype(np.float32) / 255
    except Exception as e:
        raise OSError('Error handling image at: {}'.format(fn)) from e

class DrawingDataset(FilesDataset):
    def __init__(self, fnames, y, tfm, path, classes):
        assert isinstance(classes, (list, np.ndarray)), 'classes must be label list.'
        self.y, self.c = y, len(classes)
        self.cls2id = {o:i for i, o in enumerate(classes)}
        super().__init__(fnames, tfm, path)
    def get_y(self, i): return self.cls2id[self.y[i]]
    def get_x(self, i): return open_image(os.path.join(self.path, self.fnames[i]))
    def get_c(self): return self.c

ds = ImageData.get_ds(DrawingDataset, (trn_x, trn_y), (val_x, val_y), tfms,
            test=test_x_names, path=PATH, classes=classes)
md = ImageData(PATH, ds, bs, num_workers=12, classes=classes)
md.c, md.classes[:3]
```

除了算力门槛，这个比赛的另一个要点是如何利用dataset中其他信息，如countrycode，一种方法是将它们也和drawing一起编码（encode）进图像中。

上例中的draw_cv2()只用了一个channel来创建图像，并没有用到另外两个channel：
```
img_rgb = np.repeat(img[:, :, None], 3, -1)
```
实际上，可以用不同颜色来encode drawing和countrycode，例如不同笔画用不同的颜色，为每个国家指定一种特定颜色等，构建一套可以encode
 额外信息的规则可以让模型提取到更多的特征。

```
colors = [(255, 0, 0) , (255, 255, 0),  (128, 255, 0),  (0, 255, 0), 
          (0, 255, 128), (0, 255, 255), (0, 128, 255), (0, 0, 255), 
          (128, 0, 255), (255, 0, 255)]
BASE_SIZE = 256
def draw_cv2(raw_strokes, size=256, lw=7, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE, 3), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = colors[min(t, len(colors)-1)]
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color,
                          lw, lineType=cv2.LINE_AA)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        return img

x = md.trn_ds.denorm(x)
fig, axes = plt.subplots(3, 3, figsize=(9, 6))
for i, ax in enumerate(axes.flat):
  ax.imshow(x[i])
  ax.set_title(md.classes[y[i]])
plt.tight_layout()
```
![](https://upload-images.jianshu.io/upload_images/13575947-957aa21a2a12c40b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这里的draw_cv2用不同的颜色来区分不同的笔画，如果你有兴趣，也可以用特定的颜色来表示某个countrycode或是为不同的国家在图像上添加一些特殊的标记。

## Dataset

```
r = 0.01  # a total of about 500k images
k = 1500  # a total of 510k images
sz = 128

def df_to_ims(path):
    df = pd.read_csv(path, nrows=k)
    ......
```

我在建模阶段，一般会选择小数据集以加快建模速度，这里每个分类只取1500个样本，共计51万个训练样本，数据集图像大小是128x128。有参赛者建议将大小提升到224x224、256x256可以得到更好的效果，如果你算力、时间充足可以尝试。

## MobileNet

我个人是非常推崇transfer learning的，像这类图像识别问题，首选resnet、resnext这些经Imagenet预训练的模型，但因为Quick, Draw!提供了规模庞大的数据，这让重新训练一个模型成为可能。从理论上讲，pretrained models会让模型更快收敛，但如果不考虑数据量和训练时间，目前还没有论断证明transfer learning比training from strach更好。这就好比说，让一个职业跑步运动员转行踢足球，他会比其他人更快成为职业球员，但却不一定比一个从小就花大量时间金钱练习足球的同龄球员要更强，在现实生活中，后者往往强于前者。Kaiming He他们在[Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883)提出了相同的观点。

[Mobilenet](https://arxiv.org/abs/1704.04861)是轻量的convnet，它提出一个称为Depthwise Separable Convolution（conv dw）的卷积层，即每个kernel filter只处理一个2维的freature map，而不是一个3维的freature map matrix，这样一来mobilenet的参数不仅少了很多，计算量也大幅简化。

```
def conv_dw(nin, nf, stride=2):
  return nn.Sequential(
      nn.Conv2d(nin, nin, 3, stride, 1, groups=nin, bias=False),
      ......
```
**groups=nin**，告诉conv2d()做depth-wise convolution。

```
class MobileNet(nn.Module):
  def __init__(self, num_classes, nf=512, ps=[0.25, 0.5]):
    super().__init__()
    self.layers = nn.Sequential(
        ......
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=ps[0]),
        nn.Linear(1024, nf),
        nn.ReLU(),
        nn.BatchNorm1d(nf),
        nn.Dropout(p=ps[1]),
        nn.Linear(nf, num_classes),      
        nn.LogSoftmax()
        ......
```
我在Mobilenet顶部增加两个全链接层，并添加batchnorm1d和dropout为模型增加正则化。

模型训练过程并不复杂，用[Fastai library](https://github.com/fastai/fastai/tree/master/old)提供的cyclical learning rate来训练，直到过拟合。

## 结尾

本文分享了我参加这个涂鸦识别比赛的一些心得以及starter notebook。算力是这个比赛的根本，除此之外，还要注意避免CPU和硬盘成为拉慢训练速度的短板。

## Refences

* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications](https://arxiv.org/abs/1704.04861)
* [Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883)
* [Fastai library](https://github.com/fastai/fastai/tree/master/old)









