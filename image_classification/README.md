bash benchmark.sh -p 1 -f 0 -m MobileNetV2 -b 64

-p: 是否跑paddle模型 0: 跑pytorch模型         1: 跑paddle模型
-f: 是否只跑前反向   0: 跑前反向+dataloader   1: 跑前反向
-m: 模型类型 可以为MobileNetV1 MobileNetV2 ResNet50
-b: batch_size 可以为64/128/256

