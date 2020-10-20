# ReImplement this code
by Nuller

## 1. Modify the code because of version problem
### 1. Libtorch

|old version code      |new version code    |
| ---- | ---- |
|bn_option.affline = x      |bn_option.affile(x)      |
|bn_options.stateful_ =x      |bn_option.track_running_stats(x)      |
|Conv2dOptions.with_bias(x)      |Conv2dOptions.bias(x)        |
