set -e
set -x
python -W ignore train.py lenet > lenet.out
python -W ignore train.py fc1 > fc1.out
python -W ignore train.py fc2 > fc2.out
python -W ignore train.py fc2relu > fc2relu.out
python -W ignore train.py fc2sig > fc2sig.out
python -W ignore train.py lenetRelu > lenetRelu.out
python -W ignore train.py lenetSimplify3 > lenetSimplify3.out