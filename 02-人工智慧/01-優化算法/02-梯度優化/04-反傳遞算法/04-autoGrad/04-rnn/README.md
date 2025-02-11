

## rnn.py

```
$ python rnn.py
Training RNN...
Iteration 0 Train loss: 4.854081495916476
Training text                         Predicted text
"""Implements the long-short t|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
This version vectorizes over m|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
has a fixed length."""        |zzzzzzzzzzzzzzzzzzzzzzz|zzzzzzz
from __future__ import absolut|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
from __future__ import print_f|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
from builtins import range    |zzzzzzzzzzzzzzzzzzzzzzzzzzz|zzz
import autograd.numpy as np   |zzzzzzzzzzzzzzzzzzzzzzzzzzzz|zz
import autograd.numpy.random a|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
from autograd import grad     |zzzzzzzzzzzzzzzzzzzzzzzzzz|zzzz
from autograd.scipy.special im|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
from os.path import dirname, j|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
from autograd.misc.optimizers |zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
### Helper functions #########|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
def sigmoid(x):               |zzzzzzzzzzzzzzzz|zzzzzzzzzzzzzz
    return 0.5*(np.tanh(x) + 1|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
def concat_and_multiply(weight|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    cat_state = np.hstack(args|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    return np.dot(cat_state, w|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
### Define recurrent neural ne|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
def create_rnn_params(input_si|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
                      param_sc|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    return {'init hiddens': rs|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
            'change':       rs|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
            'predict':      rs|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
def rnn_predict(params, inputs|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    def update_rnn(input, hidd|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        return np.tanh(concat_|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    def hiddens_to_output_prob|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        output = concat_and_mu|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        return output - logsum|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    num_sequences = inputs.sha|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    hiddens = np.repeat(params|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    output = [hiddens_to_outpu|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    for input in inputs:  # It|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        hiddens = update_rnn(i|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        output.append(hiddens_|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    return output             |zzzzzzzzzzzzzzzzzz|zzzzzzzzzzzz
def rnn_log_likelihood(params,|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    logprobs = rnn_predict(par|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    loglik = 0.0              |zzzzzzzzzzzzzzzzz|zzzzzzzzzzzzz
    num_time_steps, num_exampl|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    for t in range(num_time_st|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
        loglik += np.sum(logpr|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    return loglik / (num_time_|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
### Dataset setup ############|zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
Iteration 10 Train loss: 3.1128929080865673
Training text                         Predicted text
"""Implements the long-short t|    d   t t     t
This version vectorizes over m|          d d   t   d t
has a fixed length."""        |        d d   t   t d d
from __future__ import absolut|      d       d         d   t
from __future__ import print_f|      d       d           d
from builtins import range    |        d   d           d t
import autograd.numpy as np   |        d     d d t       d
import autograd.numpy.random a|        d     d d t   d d     d
from autograd import grad     |      d     d           d
from autograd.scipy.special im|      d     d d t   d   t d
from os.path import dirname, j|        d d t d       d d
from autograd.misc.optimizers |      d     d d d t
### Helper functions #########|              d   d   d d d d d
def sigmoid(x):               |      d   d d
    return 0.5*(np.tanh(x) + 1|      d t d   p d     d d
def concat_and_multiply(weight|        t   d     t d t d d
    cat_state = np.hstack(args|      d d           t   t d
    return np.dot(cat_state, w|      d t d d d   d d d
### Define recurrent neural ne|        d t   t   t   d t d   t
def create_rnn_params(input_si|        d t   d       d d t d d
                      param_sc|                        d d d t
    return {'init hiddens': rs|      d t d d d d   d   d d
            'change':       rs|              t d   d
            'predict':      rs|                t d
def rnn_predict(params, inputs|      d d     t d d d     d t
    def update_rnn(input, hidd|      d     d t   d d       d
        return np.tanh(concat_|          t t d d d d t t d d d
    def hiddens_to_output_prob|      d   d   d d       t d   p
        output = concat_and_mu|          t       t d d d d d t
        return output - logsum|          t t d     t
    num_sequences = inputs.sha|      t d t t d t     d t     d
    hiddens = np.repeat(params|      d   d           d d d d
    output = [hiddens_to_outpu|      t         d   d d       t
    for input in inputs:  # It|          d t   d d
        hiddens = update_rnn(i|          d   d       d t   d d
        output.append(hiddens_|          t     d   d d d   d d
    return output             |      d t d     t
def rnn_log_likelihood(params,|      d d   d d t d
    logprobs = rnn_predict(par|                  d   t d
    loglik = 0.0              |        t
    num_time_steps, num_exampl|      t d d t   t     t d d   t
    for t in range(num_time_st|            d   d t d       d
        loglik += np.sum(logpr|            t             t
    return loglik / (num_time_|      d t d t   d     d       d
### Dataset setup ############|        d t   t t   d d d d d d
Iteration 20 Train loss: 2.7274787709420205
Training text                         Predicted text
"""Implements the long-short t|
This version vectorizes over m|               t
has a fixed length."""        |
from __future__ import absolut|         t t                 t
from __future__ import print_f|         t t
from builtins import range    |       t
import autograd.numpy as np   |         t
import autograd.numpy.random a|         t
from autograd import grad     |       t
from autograd.scipy.special im|       t
from os.path import dirname, j|                      n
from autograd.misc.optimizers |       t
### Helper functions #########| ###         t         ########
def sigmoid(x):               |      n
    return 0.5*(np.tanh(x) + 1|        t
def concat_and_multiply(weight|                 t
    cat_state = np.hstack(args|
    return np.dot(cat_state, w|        t
### Define recurrent neural ne| ###
def create_rnn_params(input_si|
                      param_sc|
    return {'init hiddens': rs|        t           n  t
            'change':       rs|
            'predict':      rs|
def rnn_predict(params, inputs|                            t
    def update_rnn(input, hidd|         tu            t    n
        return np.tanh(concat_|            t
    def hiddens_to_output_prob|          n  t       t  t
        output = concat_and_mu|          t  t                t
        return output - logsum|            t       t        t
    num_sequences = inputs.sha|      t     t
    hiddens = np.repeat(params|      n  t
    output = [hiddens_to_outpu|         t                 t  t
    for input in inputs:  # It|                     t     #
        hiddens = update_rnn(i|          n  t
        output.append(hiddens_|          t  t
    return output             |        t       t
def rnn_log_likelihood(params,|              n
    logprobs = rnn_predict(par|
    loglik = 0.0              |
    num_time_steps, num_exampl|      t               t
    for t in range(num_time_st|
        loglik += np.sum(logpr|                       t
    return loglik / (num_time_|        t
### Dataset setup ############| ###            t  ############
Iteration 30 Train loss: 2.2877981982797126
Training text                         Predicted text
"""Implements the long-short t|    meurt tge   rt ogmon oneo
This version vectorizes over m|  on   tu ndp ettrgrnet  metu e
has a fixed length."""        |  t  t  xdtu rtporrx
from __future__ import absolut|  oge  ortototer neutrr ts grts
from __future__ import print_f|  oge  ortototer neutrr urnpo r
from builtins import range    |  oge etnronp  neutrr etgrt
import autograd.numpy as np   |  eutr  ttogrntenptsur tr pu
import autograd.numpy.random a|  eutr  ttogrntenptsurnetget  t
from autograd import grad     |  oge ttrgrnme neutrr eote
from autograd.scipy.special im|  oge ttrgrnmen onurn ottnmonne
from os.path import dirname, j|  oge minutrr neutrr en ptst
from autograd.misc.optimizers |  oge ttrgrnmen n onmurnen tn
### Helper functions #########| ### ettrtr itnrrnnp  #########
def sigmoid(x):               |  tr  n  gnenn
    return 0.5*(np.tanh(x) + 1|     eteton en*enpunstprnn
def concat_and_multiply(weight|  tr rtnrto tperstonnurrnetneor
    cat_state = np.hstack(args|     rtr  stst   penrostr nmor
    return np.dot(cat_state, w|     eteton punetrnutr  stst  e
### Define recurrent neural ne| ### etrnpt etrtostpe pttomornt
def create_rnn_params(input_si|  tr rrttoternp utrts nnputo  n
                      param_sc|                       etrts  r
    return {'init hiddens': rs|     eteton  :npne  xdenp n  s
            'change':       rs|              rrtprt         e
            'predict':      rs|              urtunton       e
def rnn_predict(params, inputs|  tr enp urtunuonutats   nputo
    def update_rnn(input, hidd|     etr tortoternpnnputo   x e
        return np.tanh(concat_|         eteton punstprnttnrto
    def hiddens_to_output_prob|     etr  xdonpir nrntouto urns
        output = concat_and_mu|         ptouto   rtnrto tperst
        return output - logsum|         eteton utouto s onrots
    num_sequences = inputs.sha|     pts  trttnrtr   nputo n on
    hiddens = np.repeat(params|     indonpi   penstuttrnutats
    output = [hiddens_to_outpu|     ptouto   uondonpir nrntout
    for input in inputs:  # It|      ns nputo np nputo    # ds
        hiddens = update_rnn(i|         indonpi   tontoternpnn
        output.append(hiddens_|         ptoutonturtpenrxdenpir
    return output             |     eteton utouto
def rnn_log_likelihood(params,|  tr enp ogrorn trnetgenutats
    logprobs = rnn_predict(par|     onronms    e_neurtunuonuta
    loglik = 0.0              |     onron    exe
    num_time_steps, num_exampl|     pts oneteistur  pts tntsur
    for t in range(num_time_st|      ns e np etprtnpts oneteis
        loglik += np.sum(logpr|         onron     pun tsnrnron
    return loglik / (num_time_|     eteton onton     sts onete
### Dataset setup ############| ### etrtrte  tstu ############
Traceback (most recent call last):
  File "D:\ccc\ccc112b\py2cs\03-人工智慧\02-優化算法\02-深度學習優化\04-反傳遞算法\04-autoGrad\04-rnn\rnn.py", line 115, in <module>
    trained_params = adam(training_loss_grad, init_params, step_size=0.1,
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\misc\optimizers.py", line 28, in _optimize
    return unflatten(optimize(_grad, _x0, _callback, *args, **kwargs))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\misc\optimizers.py", line 64, in adam
    g = grad(x, i)
        ^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\misc\optimizers.py", line 23, in <lambda>
    _grad = lambda x, i: flatten(grad(unflatten(x), i))[0]
                                 ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\wrap_util.py", line 20, in nary_f
    return unary_operator(unary_f, x, *nary_op_args, **nary_op_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\differential_operators.py", line 28, in grad
    vjp, ans = _make_vjp(fun, x)
               ^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\core.py", line 10, in make_vjp
    end_value, end_node =  trace(start_node, fun, x)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\tracer.py", line 10, in trace
    end_box = fun(start_box)
              ^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\wrap_util.py", line 15, in unary_f
    return fun(*subargs, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ccc\ccc112b\py2cs\03-人工智慧\02-優化算法\02-深度學習優化\04-反傳遞算法\04-autoGrad\04-rnn\rnn.py", line 104, in training_loss
    return -rnn_log_likelihood(params, train_inputs, train_inputs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ccc\ccc112b\py2cs\03-人工智慧\02-優化算法\02-深度學習優化\04-反傳遞算法\04-autoGrad\04-rnn\rnn.py", line 56, in rnn_log_likelihood
    loglik += np.sum(logprobs[t] * targets[t])
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\tracer.py", line 45, in f_wrapped
    node = node_constructor(ans, f_wrapped, argvals, kwargs, argnums, parents)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\core.py", line 36, in __init__
    self.vjp = vjpmaker(parent_argnums, value, args, kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\core.py", line 66, in vjp_argnums
    vjp = vjpfun(ans, *args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\numpy\numpy_vjps.py", line 298, in grad_np_sum
    shape, dtype = anp.shape(x), anp.result_type(x)
                                 ^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\tracer.py", line 61, in f_wrapped
    return f_raw(*argvals, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<__array_function__ internals>", line 177, in result_type
KeyboardInterrupt
```

## lstm.py

```
$ python lstm.py
Training LSTM...
Iteration 0 Train loss: 4.850196232440834
Training text                         Predicted text
"""Implements the long-short t|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
This version vectorizes over m|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
has a fixed length."""        |qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
from __future__ import absolut|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
from __future__ import print_f|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
from builtins import range    |qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
from os.path import dirname, j|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
import autograd.numpy as np   |qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
import autograd.numpy.random a|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
from autograd import grad     |qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
from autograd.scipy.special im|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
from autograd.misc.optimizers |qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
from rnn import string_to_one_|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
                build_dataset,|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
def init_lstm_params(input_siz|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
                     param_sca|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    def rp(*shape):           |qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
        return rs.randn(*shape|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    return {'init cells':   rp|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
            'init hiddens': rp|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
            'change':       rp|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
            'forget':       rp|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
            'ingate':       rp|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
            'outgate':      rp|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
            'predict':      rp|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
def lstm_predict(params, input|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    def update_lstm(input, hid|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
        change  = np.tanh(conc|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
        forget  = sigmoid(conc|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
        ingate  = sigmoid(conc|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
        outgate = sigmoid(conc|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
        cells   = cells * forg|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
        hiddens = outgate * np|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
        return hiddens, cells |qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    def hiddens_to_output_prob|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
        output = concat_and_mu|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
        return output - logsum|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    num_sequences = inputs.sha|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    hiddens = np.repeat(params|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    cells   = np.repeat(params|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    output = [hiddens_to_outpu|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    for input in inputs:  # It|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
        hiddens, cells = updat|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
        output.append(hiddens_|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    return output             |qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
def lstm_log_likelihood(params|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    logprobs = lstm_predict(pa|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    loglik = 0.0              |qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    num_time_steps, num_exampl|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
    for t in range(num_time_st|qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
Iteration 10 Train loss: 2.773913703548142
Training text                         Predicted text
"""Implements the long-short t|
This version vectorizes over m|
has a fixed length."""        |
from __future__ import absolut|
from __future__ import print_f|
from builtins import range    |
from os.path import dirname, j|
import autograd.numpy as np   |
import autograd.numpy.random a|
from autograd import grad     |
from autograd.scipy.special im|
from autograd.misc.optimizers |
from rnn import string_to_one_|
                build_dataset,|
def init_lstm_params(input_siz|
                     param_sca|
    def rp(*shape):           |
        return rs.randn(*shape|
    return {'init cells':   rp|
            'init hiddens': rp|
            'change':       rp|
            'forget':       rp|
            'ingate':       rp|
            'outgate':      rp|
            'predict':      rp|
def lstm_predict(params, input|
    def update_lstm(input, hid|
        change  = np.tanh(conc|
        forget  = sigmoid(conc|
        ingate  = sigmoid(conc|
        outgate = sigmoid(conc|
        cells   = cells * forg|
        hiddens = outgate * np|
        return hiddens, cells |
    def hiddens_to_output_prob|
        output = concat_and_mu|
        return output - logsum|
    num_sequences = inputs.sha|
    hiddens = np.repeat(params|
    cells   = np.repeat(params|
    output = [hiddens_to_outpu|
    for input in inputs:  # It|
        hiddens, cells = updat|
        output.append(hiddens_|
    return output             |
def lstm_log_likelihood(params|
    logprobs = lstm_predict(pa|
    loglik = 0.0              |
    num_time_steps, num_exampl|
    for t in range(num_time_st|
Iteration 20 Train loss: 2.5310719887866715
Training text                         Predicted text
"""Implements the long-short t|      uuuuuu     u uuuu   uu
This version vectorizes over m|   u    u uuu  uu uuu u  u uu u
has a fixed length."""        |        u uu uuuu   u
from __future__ import absolut|      uu u uuuuu uuuuu  uu uuu
from __future__ import print_f|      uu u uuuuu uuuuu  uuuu u
from builtins import range    |      uuuu uu  uuuuu  uuuuu
from os.path import dirname, j|      u  uu   uuuuu  uuuuuuu
import autograd.numpy as np   |    uu  uu uuuuu uuuu  u  uu
import autograd.numpy.random a|    uu  uu uuuuu uuuu  uuuuuu u
from autograd import grad     |      uu uuuuu uuuuu  uuuu
from autograd.scipy.special im|      uu uuuuu  uuu   uuuuuu uu
from autograd.misc.optimizers |      uu uuuuu uu u uu uuu uu
from rnn import string_to_one_|      uuu uuuuu    uuuuu uuuuuu
                build_dataset,|                   uuuuuu u u
def init_lstm_params(input_siz|     uuu uu  uuuuuuu uuuuu u u
                     param_sca|                         u u uu
    def rp(*shape):           |         uuu   uuuu
        return rs.randn(*shape|            uuu u  uuuuuu   uuu
    return {'init cells':   rp|        uuu uuuuu  uuuu u    uu
            'init hiddens': rp|              uuu  uuuuuu u  uu
            'change':       rp|                uuuuu        uu
            'forget':       rp|               uuuu u        uu
            'ingate':       rp|              uuuu uu        uu
            'outgate':      rp|                 uu uu       uu
            'predict':      rp|                 uuu u       uu
def lstm_predict(params, input|         uuuuuuu uuuuuu   uuuu
    def update_lstm(input, hid|          uuu uuu  uuuuuu   uuu
        change  = np.tanh(conc|           uuuu    uu  uu uuuuu
        forget  = sigmoid(conc|             u      uuuuuuuuuuu
        ingate  = sigmoid(conc|          u u u     uuuuuuuuuuu
        outgate = sigmoid(conc|             u u    uuuuuuuuuuu
        cells   = cells * forg|           uu      uuuu     uuu
        hiddens = outgate * np|          uuuuu    uu uu u   uu
        return hiddens, cells |            uuu  uuuuu   uuuu
    def hiddens_to_output_prob|          uuuuu u uuuu uu uuuuu
        output = concat_and_mu|            uu    uuuuu uuuuuuu
        return output - logsum|            uuu uu uu    uuu uu
    num_sequences = inputs.sha|        u uuuuuuu    uuuu     u
    hiddens = np.repeat(params|      uuuuu    uu uuuuu uuuuuu
    cells   = np.repeat(params|       uu      uu uuuuu uuuuuu
    output = [hiddens_to_outpu|        uu    u uuuuu u uuuu uu
    for input in inputs:  # It|         uuuu  uu uuuu       u
        hiddens, cells = updat|          uuuuu   uuuu    uuuu
        output.append(hiddens_|            uu  uuuuuuu uuuuu u
    return output             |        uuu uu uu
def lstm_log_likelihood(params|         uuuuuuu uuu uuuuuuuuu
    logprobs = lstm_predict(pa|        uuuu    u  uuuuuuuu uuu
    loglik = 0.0              |        uu    u u
    num_time_steps, num_exampl|        u uuuu  uu   uuuuuuuuuu
    for t in range(num_time_st|           uu uuuuuuuuuu uuuu
Iteration 30 Train loss: 2.4598515088414397
Training text                         Predicted text
"""Implements the long-short t|  oooeetttttt  ttt ttttt tttt t
This version vectorizes over m|  oe  ett ttt ttttttttt  tttt t
has a fixed length."""        |  o  o  tttt ttttttt
from __future__ import absolut|  ooe tt ttttttt tttttt tt tttt
from __future__ import print_f|  ooe tt ttttttt tttttt tttttt
from builtins import range    |  ooe ttttttt  tttttt ttttt
from os.path import dirname, j|  ooe t ttttt tttttt ttttttt
import autograd.numpy as np   |  ooott tttttttttttttttt  tt
import autograd.numpy.random a|  ooott ttttttttttttttttttttt t
from autograd import grad     |  ooe ttttttttttttttt tttt
from autograd.scipy.special im|  ooe ttttttttt ttttt tttttt tt
from autograd.misc.optimizers |  ooe ttttttttttt ttttttttttt
from rnn import string_to_one_|  ooe ttt tttttt  ttttttttttttt
                build_dataset,|                 oetttttttt tt
def init_lstm_params(input_siz|  o  ettttt tttttttt ttttttt tt
                     param_sca|                      ooettt tt
    def rp(*shape):           |     oo  ett  ttttt
        return rs.randn(*shape|         ootttt t ttttttt  tttt
    return {'init cells':   rp|     ootttt tttttt tttt t    tt
            'init hiddens': rp|             otttt tttttt t  tt
            'change':       rp|             ootttttt        tt
            'forget':       rp|             o ettttt        tt
            'ingate':       rp|             ottttttt        tt
            'outgate':      rp|             oottttttt       tt
            'predict':      rp|             oootttttt       tt
def lstm_predict(params, input|  o  o tttttttttttttttt   ttttt
    def update_lstm(input, hid|     oo  tttttttt tttttttt  ttt
        change  = np.tanh(conc|         oeettt    tttttttttttt
        forget  = sigmoid(conc|         ooettt     ttttttttttt
        ingate  = sigmoid(conc|         ottttt     ttttttttttt
        outgate = sigmoid(conc|         oettttt    ttttttttttt
        cells   = cells * forg|         ooet      tttt     ttt
        hiddens = outgate * np|         ottttt    ttttttt   tt
        return hiddens, cells |         ootttt tttttt   tttt
    def hiddens_to_output_prob|     oo  tttttt ttttttttttttttt
        output = concat_and_mu|         oetttt   ttttttttttttt
        return output - logsum|         ootttt tttttt   ttt tt
    num_sequences = inputs.sha|     ottt ttttttt    ttttt t tt
    hiddens = np.repeat(params|     ottttt    ttttttttttttttt
    cells   = np.repeat(params|     ooet      ttttttttttttttt
    output = [hiddens_to_outpu|     oetttt   ttttttt ttttttttt
    for input in inputs:  # It|     ooe ttttt tt ttttt      tt
        hiddens, cells = updat|         ottttt   tttt    ttttt
        output.append(hiddens_|         oetttttttttttttttttt t
    return output             |     ootttt tttttt
def lstm_log_likelihood(params|  o  o ttttttttt ttttttttttttt
    logprobs = lstm_predict(pa|     oottttt    t ttttttttttttt
    loglik = 0.0              |     oottt    ttt
    num_time_steps, num_exampl|     otttttttt ttt   tttttttttt
    for t in range(num_time_st|     ooe t tt ttttttttttttttt t
Traceback (most recent call last):
  File "D:\ccc\ccc112b\py2cs\03-人工智慧\02-優化算法\02-深度學習優化\04-反傳遞算法\04-autoGrad\04-rnn\lstm.py", line 97, in <module>
    trained_params = adam(training_loss_grad, init_params, step_size=0.1,
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\misc\optimizers.py", line 28, in _optimize
    return unflatten(optimize(_grad, _x0, _callback, *args, **kwargs))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\misc\optimizers.py", line 64, in adam
    g = grad(x, i)
        ^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\misc\optimizers.py", line 23, in <lambda>
    _grad = lambda x, i: flatten(grad(unflatten(x), i))[0]
                                 ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\wrap_util.py", line 20, in nary_f
    return unary_operator(unary_f, x, *nary_op_args, **nary_op_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\differential_operators.py", line 28, in grad
    vjp, ans = _make_vjp(fun, x)
               ^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\core.py", line 10, in make_vjp
    end_value, end_node =  trace(start_node, fun, x)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\tracer.py", line 10, in trace
    end_box = fun(start_box)
              ^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\wrap_util.py", line 15, in unary_f
    return fun(*subargs, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ccc\ccc112b\py2cs\03-人工智慧\02-優化算法\02-深度學習優化\04-反傳遞算法\04-autoGrad\04-rnn\lstm.py", line 86, in training_loss
    return -lstm_log_likelihood(params, train_inputs, train_inputs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ccc\ccc112b\py2cs\03-人工智慧\02-優化算法\02-深度學習優化\04-反傳遞算法\04-autoGrad\04-rnn\lstm.py", line 57, in lstm_log_likelihood
    logprobs = lstm_predict(params, inputs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ccc\ccc112b\py2cs\03-人工智慧\02-優化算法\02-深度學習優化\04-反傳遞算法\04-autoGrad\04-rnn\lstm.py", line 53, in lstm_predict
    output.append(hiddens_to_output_probs(hiddens))
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ccc\ccc112b\py2cs\03-人工智慧\02-優化算法\02-深度學習優化\04-反傳遞算法\04-autoGrad\04-rnn\lstm.py", line 43, in hiddens_to_output_probs
    output = concat_and_multiply(params['predict'], hiddens)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ccc\ccc112b\py2cs\03-人工智慧\02-優化算法\02-深度學習優化\04-反傳遞算法\04-autoGrad\04-rnn\rnn.py", line 22, in concat_and_multiply
    cat_state = np.hstack(args + (np.ones((args[0].shape[0], 1)),))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\numpy\numpy_wrapper.py", line 41, in hstack
    arrs = [atleast_1d(_m) for _m in tup]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\numpy\numpy_wrapper.py", line 41, in <listcomp>
    arrs = [atleast_1d(_m) for _m in tup]
            ^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\tracer.py", line 44, in f_wrapped
    ans = f_wrapped(*argvals, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\tracer.py", line 37, in f_wrapped
    boxed_args, trace, node_constructor = find_top_boxed_args(args)
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\autograd\tracer.py", line -1, in find_top_boxed_args
KeyboardInterrupt
```