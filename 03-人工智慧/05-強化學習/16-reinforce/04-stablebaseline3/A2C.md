
## 在 colab 上執行


* https://colab.research.google.com/drive/1meetmk6t5RYCDGeP_fzq7zGsP_9SfMIA?usp=sharing

colab 無法 render，因為需要有螢幕

* https://stackoverflow.com/questions/40195740/how-to-run-openai-gym-render-over-a-server

## 在 conda 上執行

```
* (base) PS D:\ccc\ai\16-reinforce\04-stablebaseline3> python A2C.py
Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
Traceback (most recent call last):
  File "D:\ccc\ai\16-reinforce\04-stablebaseline3\A2C.py", line 7, in <module>
    model = A2C("MlpPolicy", env, verbose=1)
  File "C:\Users\ccc\miniconda3\lib\site-packages\stable_baselines3\a2c\a2c.py", line 126, in __init__
    self._setup_model()
  File "C:\Users\ccc\miniconda3\lib\site-packages\stable_baselines3\common\on_policy_algorithm.py", line 123, in _setup_model
    self.policy = self.policy_class(  # pytype:disable=not-instantiable
  File "C:\Users\ccc\miniconda3\lib\site-packages\stable_baselines3\common\policies.py", line 483, in __init__
    self._build(lr_schedule)
  File "C:\Users\ccc\miniconda3\lib\site-packages\stable_baselines3\common\policies.py", line 572, in _build
    module.apply(partial(self.init_weights, gain=gain))
  File "C:\Users\ccc\miniconda3\lib\site-packages\torch\nn\modules\module.py", line 668, in apply
    module.apply(fn)
  File "C:\Users\ccc\miniconda3\lib\site-packages\torch\nn\modules\module.py", line 668, in apply
    module.apply(fn)
  File "C:\Users\ccc\miniconda3\lib\site-packages\torch\nn\modules\module.py", line 669, in apply
    fn(self)
  File "C:\Users\ccc\miniconda3\lib\site-packages\stable_baselines3\common\policies.py", line 288, in init_weights
    nn.init.orthogonal_(module.weight, gain=gain)
  File "C:\Users\ccc\miniconda3\lib\site-packages\torch\nn\init.py", line 483, in orthogonal_
    q, r = torch.linalg.qr(flattened)
RuntimeError: Calling torch.geqrf on a CPU tensor requires compiling PyTorch with LAPACK. Please use PyTorch built with LAPACK support.
```