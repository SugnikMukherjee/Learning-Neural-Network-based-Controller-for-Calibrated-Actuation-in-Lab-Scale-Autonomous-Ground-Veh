### Learning Neural Network-based Controller for Calibrated Actuation in Lab-Scale Autonomous Ground Vehicles

### *Neural Network Model Information*
---
**Model 1**:
- **Input Features**: `prev_accel_x`, `prev_accel_y`, `prev_yaw_rate`, `curr_voltage`
- **Output Features**: `throttle_pwm`, `steering_pwm`
- **TensorFlow Model**:
  - Inference time: 
    - CPU times: 
      - User: 271 ms
      - System: 7.07 ms
      - Total: 278 ms
    - Wall time: 705 ms
- **PyTorch Model**:
  - Inference time: 349 µs ± 36.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
- **TensorRT Model**:
  - Inference time: 1.15 ms ± 196 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

**Model 2**:
- **Input Features**: `curr_accel_x`, `curr_accel_y`, `curr_yaw_rate`, `curr_voltage`
- **Output Features**: `throttle_pwm`, `steering_pwm`
- **TensorFlow Model**:
  - Inference time:
    - CPU times: 
      - User: 277 ms
      - System: 2.83 ms
      - Total: 280 ms
    - Wall time: 372 ms
- **PyTorch Model**:
  - Inference time: 218 µs ± 56.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
- **TensorRT Model**:
  - Inference time: The slowest run took 221.82 times longer than the fastest. This could mean that an intermediate result is being cached.
    - 21.4 ms ± 49 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

**Model 3**:
- **Input Features**: `prev_accel`, `prev_yaw_rate`, `curr_voltage`
- **Output Features**: `throttle_pwm`, `steering_pwm`
- **PyTorch Model**:
  - Inference time: 279 µs ± 92.6 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
- **TensorRT Model**:
  - Inference time: The slowest run took 385.24 times longer than the fastest. This could mean that an intermediate result is being cached.
    - 32.7 ms ± 78.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


**Model 4**:
- **Input Features**: `curr_accel`, `curr_yaw_rate`, `curr_voltage`
- **Output Features**: `throttle_pwm`, `steering_pwm`
- **PyTorch Model**:
  - Inference time: 231 µs ± 55.2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
- **TensorRT Model**:
  - Inference time: 999 µs ± 422 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

**Model 5 using simple ANN**:
- **Input Features**: `prev_accel_x`,`prev_accel_y`,`prev_yaw_rate`, `curr_voltage`,`curr_accel_x`,`curr_accel_y`,`curr_yaw_rate`
- **Output Features**: `throttle_pwm`, `steering_pwm`
- **PyTorch Model**:
  - Inference time: 304 µs ± 17.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
- **TensorRT Model**:
  - Inference time: The slowest run took 47.47 times longer than the fastest. This could mean that an intermediate result is being cached.
    - 4.63 ms ± 8.86 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


**Model 5 using LSTM**:
- **Input Features**: `prev_accel_x`,`prev_accel_y`,`prev_yaw_rate`, `curr_voltage`,`curr_accel_x`,`curr_accel_y`,`curr_yaw_rate`
- **Output Features**: `throttle_pwm`, `steering_pwm`
- **PyTorch Model** (Using LSTM):
  - Inference time: 2 ms ± 150 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


**Model 5 using GRU**:
- **Input Features**: `prev_accel_x`,`prev_accel_y`,`prev_yaw_rate`, `curr_voltage`,`curr_accel_x`,`curr_accel_y`,`curr_yaw_rate`
- **Output Features**: `throttle_pwm`, `steering_pwm`
- **PyTorch Model** (Using GRU):
  - Inference time: 2.54 ms ± 228 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


**Model 6**:
- (Same as Model 5 with new training data)
- **Input Features**: `prev_accel_x`,`prev_accel_y`,`prev_yaw_rate`, `curr_voltage`,`curr_accel_x`,`curr_accel_y`,`curr_yaw_rate`
- **Output Features**: `throttle_pwm`, `steering_pwm`
- **PyTorch Model**:
  - Inference time: 274 µs ± 69 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

**Model 7 using DNN**:
- **Input Features**: `prev_accel`,`prev_yaw_rate`, `curr_voltage`,`curr_accel`,`curr_yaw_rate`
- **Output Features**: `throttle_pwm`, `steering_pwm`
- **PyTorch Model**:
  - Inference time: 254 µs ± 76.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
- **TensorRT Model**:
  - Inference time: The slowest run took 47.47 times longer than the fastest. This could mean that an intermediate result is being cached.
    - 4.63 ms ± 8.86 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

**Model 7 using LSTM**:
- **Input Features**: `prev_accel`,`prev_yaw_rate`, `curr_voltage`,`curr_accel`,`curr_yaw_rate`
- **Output Features**: `throttle_pwm`, `steering_pwm`
- **PyTorch Model**:
  - Inference time: 1.4 ms ± 248 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

**Model 7 using GRU**:
- **Input Features**: `prev_accel`,`prev_yaw_rate`, `curr_voltage`,`curr_accel`,`curr_yaw_rate`
- **Output Features**: `throttle_pwm`, `steering_pwm`
- **PyTorch Model**:
  - Inference time: 1.6 ms ± 347 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
