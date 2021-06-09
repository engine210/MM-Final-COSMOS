
Root directory of this project is `~/MMFinal`

Create a virtualenv and activate it first

## Clone the repo
```sh
cd ~/MMFinal
git clone https://github.com/shivangi-aneja/COSMOS.git
git clone https://github.com/facebookresearch/detectron2.git
```

## COSMOS
```sh
cd COSMOS
chcuda 11.0
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m spacy download en
```

## detectron2
```sh
cd ~/MMFinal/detectron2
git checkout 4841e70 # checkout to v0.3
```
Replace files
```sh
cd ~/MMFinal
cp COSMOS/detectron2_changes/config/defaults.py detectron2/detectron2/config
cp COSMOS/detectron2_changes/engine/defaults.py detectron2/detectron2/engine
cp COSMOS/detectron2_changes/modeling/meta_arch/rcnn.py detectron2/detectron2/modeling/meta_arch
```
Install detectron2
```sh
cd ~/MMFinal
pip install -e detectron2
```

## Change some code
`utils/config.py`
```py
# Data Directories
BASE_DIR = '/home/engine210/MMFinal'
DATA_DIR = '/warehouse/engine210/MMData'
TARGET_DIR = "/home/engine210/MMFinal/viz"
```

`trainer_scipt.py`
```py
# DataLoaders
train_dataset = CaptionInContext(metadata_file=os.path.join(DATA_DIR, 'mmsys_anns', 'train_data_delete_bad.json'),
                                 transforms=img_transform_train, mode='train', text_field=text_field)

val_dataset = CaptionInContext(metadata_file=os.path.join(DATA_DIR, 'mmsys_anns', 'val_data.json'),
                               transforms=img_transform, mode='val', text_field=text_field)

test_dataset = CaptionInContext(metadata_file=os.path.join(DATA_DIR, 'mmsys_anns', 'public_test_mmsys_final.json'),
                               transforms=img_transform, mode='test', text_field=text_field)
```

## Bugs
### Numpy
Bug log
```
Traceback (most recent call last):
  File "trainer_scipt.py", line 8, in <module>
    from utils.logging.tf_logger import Logger
  File "/home/engine210/MMFinal/COSMOS/utils/__init__.py", line 6, in <module>
    from .img_model_utils import *
  File "/home/engine210/MMFinal/COSMOS/utils/img_model_utils.py", line 2, in <module>
    from detectron2.structures import (Boxes, BoxMode, Instances)
  File "/home/engine210/MMFinal/detectron2/detectron2/structures/__init__.py", line 7, in <module>
    from .masks import BitMasks, PolygonMasks, rasterize_polygons_within_box, polygons_to_bitmask
  File "/home/engine210/MMFinal/detectron2/detectron2/structures/masks.py", line 6, in <module>
    import pycocotools.mask as mask_util
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/pycocotools/mask.py", line 3, in <module>
    import pycocotools._mask as _mask
  File "pycocotools/_mask.pyx", line 1, in init pycocotools._mask
ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
```
Solution
```sh
pip uninstall numpy
pip install numpy
```
You can ignore the error below while reinstalling numpy
```txt
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.3.0 requires numpy<1.19.0,>=1.16.0, but you have numpy 1.20.3 which is incompatible.
```

### GPU OOM
If at the beginning of training, gpu out of memory happen like below
```
Total Params 2559576
Img Model 2405676
Text Model 153900
Loading Saved Model
  0%|                                                                                                                                                                                                                                                                                              | 0/2528 [00:00<?, ?it/s]2021-06-10 00:46:44.740380: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
  0%|                                                                                                                                                                                                                                                                                    | 1/2528 [00:06<4:48:53,  6.86s/it]Traceback (most recent call last):
  File "trainer_scipt.py", line 232, in <module>
    train_joint_model()
  File "trainer_scipt.py", line 156, in train_joint_model
    train_model(epoch)
  File "trainer_scipt.py", line 85, in train_model
    z_img, z_t_match, z_t_diff = combined_model(img, text_match, text_diff, batch, seq_len_match, seq_len_diff,
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/engine210/MMFinal/COSMOS/model_archs/models.py", line 51, in forward
    img = self.maskrcnn_extractor(img, bboxes, bbox_classes)
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/engine210/MMFinal/COSMOS/model_archs/image/image_models.py", line 47, in forward
    [self.maskrcnn_predictor(img.cpu().numpy(), target) for img, target in zip(imgs, targets)])
  File "/home/engine210/MMFinal/COSMOS/model_archs/image/image_models.py", line 47, in <listcomp>
    [self.maskrcnn_predictor(img.cpu().numpy(), target) for img, target in zip(imgs, targets)])
  File "/home/engine210/MMFinal/detectron2/detectron2/engine/defaults.py", line 256, in __call__
    predictions = self.model([inputs])
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/engine210/MMFinal/detectron2/detectron2/modeling/meta_arch/rcnn.py", line 149, in forward
    return self.inference(batched_inputs)
  File "/home/engine210/MMFinal/detectron2/detectron2/modeling/meta_arch/rcnn.py", line 203, in inference
    features = self.backbone(images.tensor)
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/engine210/MMFinal/detectron2/detectron2/modeling/backbone/resnet.py", line 436, in forward
    x = stage(x)
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/nn/modules/container.py", line 117, in forward
    input = module(input)
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/engine210/MMFinal/detectron2/detectron2/modeling/backbone/resnet.py", line 202, in forward
    out = self.conv3(out)
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/engine210/MMFinal/detectron2/detectron2/layers/wrappers.py", line 80, in forward
    x = self.norm(x)
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/engine210/MMFinal/detectron2/detectron2/layers/batch_norm.py", line 57, in forward
    return F.batch_norm(
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/nn/functional.py", line 2014, in batch_norm
    return torch.batch_norm(
RuntimeError: CUDA out of memory. Tried to allocate 240.00 MiB (GPU 0; 31.75 GiB total capacity; 690.95 MiB already allocated; 73.50 MiB free; 914.00 MiB reserved in total by PyTorch)
```
Solution:
```sh
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### Open too many files
```
Total Params 2559576
Img Model 2405676
Text Model 153900
Loading Saved Model
  0%|                                                                                                                                                                                                                                                                                              | 0/2528 [00:00<?, ?it/s]2021-06-10 00:49:13.559914: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
  1%|██▏                                                                                                                                                                                                                                                                                  | 20/2528 [00:32<54:24,  1.30s/it]Traceback (most recent call last):
  File "trainer_scipt.py", line 232, in <module>
    train_joint_model()
  File "trainer_scipt.py", line 156, in train_joint_model
    train_model(epoch)
  File "trainer_scipt.py", line 80, in train_model
    for batch_idx, (img, text_match, text_diff, seq_len_match, seq_len_diff, bboxes, bbox_classes) in enumerate(
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/tqdm/_tqdm.py", line 1000, in __iter__
    for obj in iterable:
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 363, in __next__
    data = self._next_data()
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 974, in _next_data
    idx, data = self._get_data()
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 941, in _get_data
    success, data = self._try_get_data()
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 779, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/home/engine210/.pkg/python-3.8/lib/python3.8/multiprocessing/queues.py", line 116, in get
    return _ForkingPickler.loads(res)
  File "/home/engine210/MMFinal/venv/lib/python3.8/site-packages/torch/multiprocessing/reductions.py", line 282, in rebuild_storage_fd
    fd = df.detach()
  File "/home/engine210/.pkg/python-3.8/lib/python3.8/multiprocessing/resource_sharer.py", line 58, in detach
    return reduction.recv_handle(conn)
  File "/home/engine210/.pkg/python-3.8/lib/python3.8/multiprocessing/reduction.py", line 189, in recv_handle
    return recvfds(s, 1)[0]
  File "/home/engine210/.pkg/python-3.8/lib/python3.8/multiprocessing/reduction.py", line 164, in recvfds
    raise RuntimeError('received %d items of ancdata' %
RuntimeError: received 0 items of ancdata
```
Solution:
```sh
sudo sh -c "ulimit -n 65535 && exec su $LOGNAME"
```