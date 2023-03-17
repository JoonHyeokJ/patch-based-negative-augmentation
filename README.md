# negative-augmentation
__Unofficial__ implement of negative augmentations in "Understanding and Improving Robustness of Vision Transformers through Patch-based Negative Augmentation"

## Arguments
If you want to check whole arguments when it comes to argparse, please open 3 py files (p-shuffle.py, p-rotate.py, and p-infill.py), and see some lines starting with parser below ```if __name__=="__main__"```

## Examples of executing
### P-Shuffle
```
python p-shuffle.py --img_path ./test_img/real_033_000211.jpg --save_dir ./runs/p-shuffle --result_name shuffle1.jpg
python p-shuffle.py --img_path ./test_img/real_033_000215.jpg --save_dir ./runs/p-shuffle --result_name shuffle2.jpg
```

### P-Rotate
```
python p-rotate.py --img_path ./test_img/real_033_000211.jpg --save_dir ./runs/p-rotate --result_name rotate1.jpg
python p-rotate.py --img_path ./test_img/real_033_000215.jpg --save_dir ./runs/p-rotate --result_name rotate2.jpg
```

### P-Infill
When replace rate is __0.25__
```
python p-infill.py --img_path ./test_img/real_033_000211.jpg --save_dir ./runs/p-infill --replace_rate 0.25 --result_name infill1.jpg
python p-infill.py --img_path ./test_img/real_033_000215.jpg --save_dir ./runs/p-infill --replace_rate 0.25 --result_name infill2.jpg
```
When replace rate is __0.375__
```
python p-infill.py --img_path ./test_img/real_033_000211.jpg --save_dir ./runs/p-infill --replace_rate 0.375 --result_name infill1.jpg
python p-infill.py --img_path ./test_img/real_033_000215.jpg --save_dir ./runs/p-infill --replace_rate 0.375 --result_name infill2.jpg
```