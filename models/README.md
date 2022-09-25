# Model

## cvt_io_v1_1 :

latent array： N = 10，L = 6，self attention shape：[b, n, d, H, W]

100K steps, lr: 4e-3, wd: 1e-7

## cvt_io_v2_1 :

latent array： N = 10，L = 6，self attention shape：[b, d, H, W]

100K steps, lr: 4e-3, wd: 1e-7

|             | **IOU@0.50** | **Params** | **FLOPs** | **Cost-Time** |
| ----------- | ------------ | ---------- | --------- | ------------- |
| cvt         | 0.37         | 670.32K    | 6.934G    | 58.62ms       |
| cvt_io_v1_1 | 0.34         | 936.94K    | 7.438G    | 36.91ms       |