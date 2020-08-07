# Human Det

A Single-Human Detector runs at **70 FPS** on GV100.

## Install

```
# via pip
pip install git+https://github.com/Project-Splinter/human_det --upgrade

# via git clone
git clone https://github.com/Project-Splinter/human_det
cd human_det
python setup.py develop
```

Note to run `demo.py`, you also need to install [streamer_pytorch](https://github.com/Project-Splinter/streamer_pytorch) through:
```
pip install git+https://github.com/Project-Splinter/streamer_pytorch --upgrade
```

## Usage

```
# images
python demo.py --images <IMAGE_PATH> <IMAGE_PATH> <IMAGE_PATH> --loop --vis
# image folder
python demo.py --image_folder <IMAGE_FOLDER_PATH> --loop --vis
# videos
python demo.py --videos <VIDEO_PATH> <VIDEO_PATH> <VIDEO_PATH> --vis
# capture device
python demo.py --camera --vis
```

## API
```
det_engine = Detection(device="cuda:0", fp16=False)
tensor = det_engine.load_images(image_files)
topk_bboxes, topk_probs = det_engine(tensor, class_names=["person"], topk=1)
```
**Note**: `Detection` is not an instance of `nn.Module`, so it won't be trained and updated at all.
