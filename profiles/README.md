### Effect of FileVideoStream with `queue_size=250` on performance

| Method with FVS | Frames   | Seconds  | FPS  |
| --------------- | -------- | -------- | ---- |
| No              | 500      | 6.19     | 80.8 |
| Yes             | 500      | 6.44     | 77.7 |
| No              | 33838    | 439.1    | 77.1 |
| Yes             | 33838    | 432.2    | 78.3 |


### Replaced Pandas with Numpy and HDF5
| Method with FVS | Frames   | Seconds  | FPS   |
| --------------- | -------- | -------- | ----- |
| No              | 33838    | 330.29   | 102.4 |
| Yes (250)       | 33838    | 316.97   | 106.8 |
| Yes (100)       | 33838    | 317.68   | 106.5 |
| Yes (50)        | 33838    | 323.23   | 104.7 |
| Yes (500)       | 33838    | 327.39   | 103.4 |

