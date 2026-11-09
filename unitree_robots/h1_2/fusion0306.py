import math

def imu_read_data(acc, gyro):
    return {
        'ax': acc.x * 0.001 * 9.80665,
        'ay': acc.y * 0.001 * 9.80665,
        'az': acc.z * 0.001 * 9.80665,
        'gx': gyro.x * 0.001,
        'gy': gyro.y * 0.001,
        'gz': gyro.z * 0.001
    }


_euler_state = {
    'roll': 0.0,
    'pitch': 0.0,
    'yaw': 0.0
}

def imu_update_euler(data, dt):
    global _euler_state
    
    _euler_state['roll'] += data['gx'] * dt
    _euler_state['pitch'] += data['gy'] * dt
    _euler_state['yaw'] += data['gz'] * dt
    
    roll_acc = math.atan2(data['ay'], data['az']) * 57.29578
    pitch_acc = math.atan2(-data['ax'], math.sqrt(data['ay']**2 + data['az']**2)) * 57.29578
    
    alpha = 0.98
    
    _euler_state['roll'] = alpha * _euler_state['roll'] + (1.0 - alpha) * roll_acc
    _euler_state['pitch'] = alpha * _euler_state['pitch'] + (1.0 - alpha) * pitch_acc
    
    return _euler_state.copy()