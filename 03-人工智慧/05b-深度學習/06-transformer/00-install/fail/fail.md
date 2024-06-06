
## 訓練 -- 失敗

```
(env) mac020:11-bert mac020$ export TRAIN_FILE=dataset/wikitext2-raw/wiki.train.raw
(env) mac020:11-bert mac020$ export TEST_FILE=dataset/wikitext2-raw/wiki.test.raw
(env) mac020:11-bert mac020$ python run_language_modeling.py \
>     --output_dir=output \
>     --model_type=gpt2 \
>     --model_name_or_path=gpt2 \
>     --do_train \
>     --train_data_file=$TRAIN_FILE \
>     --do_eval \
>     --eval_data_file=$TEST_FILE
/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/Resources/Python.app/Contents/MacOS/Python: can't open file 'run_language_modeling.py': [Errno 2] No such file or directory
(env) mac020:11-bert mac020$ pwd
/Users/mac020/Desktop/ccc/ai2/python/11-deepLearning/11-bert
(env) mac020:11-bert mac020$ cd 01-install/
(env) mac020:01-install mac020$ python run_language_modeling.py \>     --output_dir=output \>     --model_type=gpt2 \>     --model_name_or_path=gpt2 \
>     --do_train \
>     --train_data_file=$TRAIN_FILE \
>     --do_eval \
>     --eval_data_file=$TEST_FILE
Neither PyTorch nor TensorFlow >= 2.0 have been found.Models won't be available and only tokenizers, configurationand file/data utilities can be used.
Traceback (most recent call last):
  File "run_language_modeling.py", line 29, in <module>
    from transformers import (
ImportError: cannot import name 'MODEL_WITH_LM_HEAD_MAPPING' from 'transformers' (/Users/mac020/Desktop/ccc/ai2/python/11-deepLearning/11-bert/env/lib/python3.7/site-packages/transformers/__init__.py)
(env) mac020:01-install mac020$ 
```

