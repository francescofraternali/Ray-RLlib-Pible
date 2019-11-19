from Pible_parameters import *
import numpy as np

def Energy(SC_volt, light, action):

    time_passed = 60
    event = 0

    Energy_Rem = SC_volt * SC_volt * 0.5 * SC_size

    if SC_volt <= SC_volt_min: # Node is death and not consuming energy
        Energy_Used = 0
    else: # Node is alive
        Energy_Used = (SC_volt * i_sens) * time_sens # Energy used to sense sensors (i.e. light)

        time_sleep = time_passed - time_sens
        '''
        if dic[str(x)] == 'PIR_events_time':
            Energy_Used += (SC_volt * i_PIR_detect) * time_PIR_detect
            time_sleep -= time_PIR_detect
        if dic[str(x)] == 'state_trans': # Energy used to send a data
            Energy_Used += (time_BLE_sens * SC_volt * i_BLE_sens) # energy consumed by the node to send data
            time_sleep -= time_BLE_sens
        '''
        # finally add time that was sleeping
        if action == 1:
            i_sl = i_sleep_PIR
        else:
            i_sl = i_sleep

        Energy_Used += (time_sleep * SC_volt * i_sl) # Energy Consumed by the node in sleep mode

        Energy_Used += (time_BLE_sens * SC_volt * i_BLE_sens) # energy consumed by the node to send data
        #Energy_Used += (PIR * I_PIR_detect * PIR_detect_time) # energy consumed by the node for the PIR

    Energy_Prod = time_passed * p_solar_1_lux * light

    # Energy cannot be lower than 0
    Energy_Rem = max(Energy_Rem - Energy_Used + Energy_Prod, 0)

    SC_volt = np.sqrt((2*Energy_Rem)/SC_size)

    # Setting Boundaries for Voltage
    if SC_volt > SC_volt_max:
        SC_volt = np.array([SC_volt_max])

    if SC_volt < SC_volt_min:
        SC_volt = np.array([SC_volt_min])

    #SC_volt = np.round(SC_volt, 4)

    return SC_volt, time_passed, event

def event_func(action, cur_pos, time_on):
        events = []
        events.append(1000)
        events.append(2000)

        t = cur_pos[0]
        for check in events:
            if t <= check and check <= t + time_on:
                events.remove(check)
                event = 1
                break
            else:
                event = 0

        return event

def reward_func(action, event, SC_volt):
    reward = 0
    if action == 1 and event == 1:
        reward = 1
    elif action == 0 and event == 1:
        reward = -1

    if SC_volt <= SC_volt_die:
        reward = -300

    return reward

def light_env(time):
    #print(time)
    if time > 8 and time < 16:
        light = 250
    else:
        light = 0

    return light

'''
def event_func(action, cur_pos, t_gran):

        if action == 0 and cur_pos[0] <= 5*t_gran:
            reward = 1
        elif action == 1 and cur_pos[0] >= 6*t_gran and cur_pos[0] <= 10*t_gran:
            reward = 1
        elif action == 0 and cur_pos[0] >= 11*t_gran and cur_pos[0] <= 12*t_gran:
            reward = 1
        elif action == 1 and cur_pos[0] >= 13*t_gran and cur_pos[0] <= 14*t_gran:
            reward = 1
        elif action == 0 and cur_pos[0] >= 15*t_gran and cur_pos[0] <= 16*t_gran:
            reward = 1
        elif action == 1 and cur_pos[0] >= 17*t_gran and cur_pos[0] <= 19*t_gran:
            reward  = 1
        elif action == 0 and cur_pos[0] >= 20*t_gran and cur_pos[0] <= 21*t_gran:
            reward = 1
        elif action == 1 and cur_pos[0] >= 22*t_gran and cur_pos[0] <= 23*t_gran:
            reward = 1
        elif action == 0 and cur_pos[0] >= 24*t_gran:
            reward = 1

        return event
        '''
