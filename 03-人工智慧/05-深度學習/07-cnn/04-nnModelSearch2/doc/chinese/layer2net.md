# Layer 描述轉 network

1. 使用 model.layers 描述層次，根據這個描述創建出 torch 的 network

```py
        self.model.layers = [
            {"type": "flatten", inShape: [28, 28] }, # outshape 自動變為 [28*28]
            {"type": "linear", inShape: [28*28], outShape: [50] },
            {"type": "relu" }, # relu 大小自動使用前面的 outShape
            {"type": "linear", outshape:[10] } # 沒指定 inShape, 預設使用前面的 outShape 
        ]
        self.net = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear([28*28], 50),
            nn.ReLU(),
            nn.Linear(50, [10])
        )
```

對於捲積層，可以參考 convnet.js 的作法

* https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html

```js
layer_defs = [];
layer_defs.push({type:'input', out_sx:24, out_sy:24, out_depth:1});
layer_defs.push({type:'conv', sx:5, filters:8, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:3, stride:3});
layer_defs.push({type:'softmax', num_classes:10});

net = new convnetjs.Net();
net.makeLayers(layer_defs);

trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:20, l2_decay:0.001});
```
