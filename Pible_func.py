from Pible_parameters import *
import numpy as np

def Energy(SC_volt, light, action, next_wake_up_time, event):

    next_wake_up_time *= 60 # in seconds

    Energy_Rem = SC_volt * SC_volt * 0.5 * SC_size

    if SC_volt <= SC_volt_min: # Node is death and not consuming energy
        Energy_Used = 0
    else: # Node is alive
        Energy_Used = (SC_volt * i_sens) * time_sens # Energy used to sense sensors (i.e. light)
        time_sleep = next_wake_up_time - time_sens

        if action == 1: # Node was able to detect events using the PIRand hence he will consume energy
            i_sl = i_sleep_PIR
            Energy_Used += (time_PIR_detect * SC_volt * i_PIR_detect) * event
            time_sleep -= event * time_PIR_detect
        else:
            i_sl = i_sleep

        Energy_Used += (time_BLE_sens * SC_volt * i_BLE_sens) # energy consumed by the node to send data
        time_sleep -= time_BLE_sens

        Energy_Used += (time_sleep * SC_volt * i_sl) # Energy Consumed by the node in sleep mode

    Energy_Prod = next_wake_up_time * p_solar_1_lux * light
    #print(Energy_Prod, Energy_Used, Energy_Rem, SC_volt, event)

    # Energy cannot be lower than 0
    Energy_Rem = max(Energy_Rem - Energy_Used + Energy_Prod, 0)

    SC_volt = np.sqrt((2*Energy_Rem)/SC_size)

    # Setting Boundaries for Voltage
    if SC_volt > SC_volt_max:
        SC_volt = np.array([SC_volt_max])
    if SC_volt < SC_volt_min:
        SC_volt = np.array([SC_volt_min])

    #SC_volt = np.round(SC_volt, 4)

    return SC_volt

def event_func(t_min, next_wake_up_time, events): # check how many events are on this laps of time
        event = 0
        for check in events:
            if t_min <= check and check <= t_min + next_wake_up_time:
                event += 1
                events.remove(check)
        #print(t_min, check, next_wake_up_time)
        return event, events

def reward_func(action, event, SC_volt):
    reward = 0
    if action == 1 and event != 0:
        reward = event
    elif action == 0 and event != 0:
        reward = -event

    if SC_volt <= SC_volt_die:
        reward = -300

    return reward

def light_env(time):
    light = 0
    if time > 8 and time < 16:
        light = 1500

    return light

def events():
    events = [100, 200, 400, 456, 542] # in minutes
    events.append(50)
    events.append(30)
    return events


def plot_hist(Time, Light, Action, Reward, Perf, SC_Volt, SC_Norm, PIR, episode, tot_rew):

    print("Total reward: ", tot_rew)
    #Start Plotting
    plt.figure(1)
    plt.subplot(411)
    plt.title(('Sensing every {0} sec, PIR {1} ({2} events). Tot reward: {3}').format(state_trans, using_PIR, PIR_events, tot_rew))
    plt.plot(Time, Light, 'b-', label = 'Light', markersize = 10)
    plt.ylabel('Light [lux]', fontsize=15)
    plt.legend(loc=9, prop={'size': 10})
    plt.ylim(0)
    plt.grid(True)
    plt.subplot(412)
    plt.plot(Time, PIR, 'k.', label = 'PIR detection', markersize = 15)
    plt.ylabel('PIR [boolean]', fontsize=15)
    plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc=9, prop={'size': 10})
    plt.ylim(-0.25, 1.25)
    plt.grid(True)
    plt.subplot(413)
    plt.plot(Time, SC_Volt, 'r.', label = 'SC Voltage', markersize = 15)
    plt.ylabel('Super Capacitor\nVoltage [V]', fontsize=15)
    plt.legend(loc=9, prop={'size': 10})
    plt.ylim(2.2,5.6)
    plt.grid(True)
    plt.subplot(414)
    plt.plot(Time, Action, 'y.', label = 'Actions', markersize = 15)
    plt.ylabel('Actions [num]', fontsize=15)
    plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc=9, prop={'size': 10})
    plt.ylim(0)
    plt.grid(True)
    plt.savefig('Graph.png', bbox_inches='tight')
    plt.show()

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
