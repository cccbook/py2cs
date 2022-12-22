## Case

start: {'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'Flatten'}], 'parameter_count': 7850, 'accuracy': 91.79000091552734} height=91.782151
1:{'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'Conv2d', 'out_channels': 32}, {'type': 'Flatten'}], 'parameter_count': 216650, 'accuracy': 92.05999755859375} height=91.843348
8:{'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'Conv2d', 'out_channels': 32}, {'type': 'ReLU'}, {'type': 'Flatten'}], 'parameter_count': 216650, 'accuracy': 97.36000061035156} height=97.143351
9:{'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'ConvPool2d', 'out_channels': 8}, {'type': 'Conv2d', 'out_channels': 32}, {'type': 'ReLU'}, {'type': 'Flatten'}], 'parameter_count': 41146, 'accuracy': 97.27999877929688} height=97.238853
12:{'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'ConvPool2d', 'out_channels': 8}, {'type': 'Conv2d', 'out_channels': 32}, {'type': 'ReLU'}, {'type': 'ConvPool2d', 'out_channels': 32}, {'type': 'Flatten'}], 'parameter_count': 16794, 'accuracy': 97.69999694824219} height=97.683203
14:{'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'ConvPool2d', 'out_channels': 2}, {'type': 'Conv2d', 'out_channels': 32}, {'type': 'ReLU'}, {'type': 'ConvPool2d', 'out_channels': 32}, {'type': 'Flatten'}], 'parameter_count': 15006, 'accuracy': 97.7699966430664} height=97.754991
15:{'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'Conv2d', 'out_channels': 32}, {'type': 'Conv2d', 'out_channels': 32}, {'type': 'ReLU'}, {'type': 'ConvPool2d', 'out_channels': 32}, {'type': 'Flatten'}], 'parameter_count': 57546, 'accuracy': 98.2699966430664} height=98.212451


## Case

start: {'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'Flatten'}], 'parameter_count': 7850, 'accuracy': 91.79000091552734} height=91.782151
1:{'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'ReLU'}, {'type': 'Flatten'}], 'parameter_count': 7850, 'accuracy': 91.91000366210938} height=91.902154
5:{'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'ConvPool2d', 'out_channels': 8}, {'type': 'Flatten'}], 'parameter_count': 13610, 'accuracy': 92.08000183105469} height=92.066392
9:{'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'ConvPool2d', 'out_channels': 8}, {'type': 'ReLU'}, {'type': 'Flatten'}], 'parameter_count': 13610, 'accuracy': 93.37000274658203} height=93.356393
40:{'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'Conv2d', 'out_channels': 4}, {'type': 'ReLU'}, {'type': 'Flatten'}], 'parameter_count': 27090, 'accuracy': 95.9800033569336} height=95.952913
47:{'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'Conv2d', 'out_channels': 4}, {'type': 'Conv2d', 'out_channels': 8}, {'type': 'ReLU'}, {'type': 'Flatten'}], 'parameter_count': 46426, 'accuracy': 97.63999938964844} height=97.593573
49:{'in_shape': [28, 28], 'out_shape': [10], 'layers': [{'type': 'Conv2d', 'out_channels': 4}, {'type': 'Conv2d', 'out_channels': 8}, {'type': 'Conv2d', 'out_channels': 32}, {'type': 'ReLU'}, {'type': 'Flatten'}], 'parameter_count': 157562, 'accuracy': 98.05000305175781} height=97.892441
 