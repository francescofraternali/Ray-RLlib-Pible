from Pible_parameters import *
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime
from time import sleep
import random
#from training_pible import light_divider

def Energy(SC_volt, light, action, next_wake_up_time, event):

    next_wake_up_time_sec = next_wake_up_time * 60 # in seconds

    Energy_Rem = SC_volt * SC_volt * 0.5 * SC_size

    if SC_volt <= SC_volt_min: # Node is death and not consuming energy
        Energy_Used = 0
    else: # Node is alive
        Energy_Used = (SC_volt * i_sens) * time_sens # Energy used to sense sensors (i.e. light)
        time_sleep = next_wake_up_time_sec - time_sens

        if action == 1: # Node was able to detect events using the PIR and hence he will consume energy
            i_sl = i_sleep_PIR
            Energy_Used += (time_PIR_detect * SC_volt * i_PIR_detect) * event # energy consumed to detect people
            time_sleep = time_sleep - (time_PIR_detect * event)
            Energy_Used += (time_BLE_sens * SC_volt * i_BLE_sens) * event # energy consumed by the node to send data
            time_sleep -= time_BLE_sens * event
        else:
            i_sl = i_sleep

        if event == 0: # Every time it wakes up there is at least one BLE communication, even with events = 0
            Energy_Used += (time_BLE_sens * SC_volt * i_BLE_sens) # energy consumed by the node to send one data
            time_sleep -= time_BLE_sens

        Energy_Used += (time_sleep * SC_volt * i_sl) # Energy Consumed by the node in sleep mode

    Energy_Prod = next_wake_up_time_sec * p_solar_1_lux * light
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

    return SC_volt, Energy_Prod, Energy_Used

def light_event_func(t_now, next_wake_up_time, count, len, light_prev, file_data, days_repeat, diff_days, light_div): # check how many events are on this laps of time
        event = 0
        light = light_prev
        #with open(file_data, 'r') as f:
        #    file = f.readlines()
        for i in range(count, len):
            line = file_data[count].split("|")
            #light_test = line[8]
            check_time_old = datetime.datetime.strptime(line[0], '%m/%d/%y %H:%M:%S')
            check_time = check_time_old + datetime.timedelta(days=days_repeat*diff_days)

            #print(t_now, check_time, next_wake_up_time, t_now.time(),  check_time.time(), next_wake_up_time.time())
            #print(t_now, check_time, next_wake_up_time, check_time_old, days_repeat, diff_days)
            #sleep(2)
            if t_now <= check_time and check_time <= next_wake_up_time:
                light = int(int(line[8])/light_div)
                PIR = int(line[6])
                event += PIR
                count += 1
            elif check_time > next_wake_up_time:
                break
            else: # check_time < t_now:
                count += 1
        return light, count, event

def reward_func(action, event, SC_volt):
    reward = 0
    if action == 1 and event != 0:
        reward = 0.1 #*event
    elif action == 0 and event != 0:
        reward = -0.1 #*(event)
    #elif action == 1 and event == 0:
    #    reward = -0.1

    if SC_volt <= SC_volt_die:
        reward = -1 #-1

    return reward

def randomize_light_time(input_data_raw):
    input_data = []
    rand_time = random.randrange(-15, 15, 1)
    rand_light = random.randrange(-30, 30, 1)
    #rand_time = 0
    #rand_light = 0
    for i in range(0,len(input_data_raw)):
        line = input_data_raw[i].split("|")
        curr_time = datetime.datetime.strptime(line[0], '%m/%d/%y %H:%M:%S')
        curr_time = curr_time + datetime.timedelta(minutes=rand_time)
        curr_time_new = curr_time.strftime('%m/%d/%y %H:%M:%S')
        light = int(line[8])
        new_light = int(light + ((light/100) * rand_light))
        #if i == 1:
        #    print(rand_light, i, light, new_light)
        #    sleep(5)
        if new_light < 0:
            new_light = 0
        line[0] = curr_time_new
        line[8] = str(new_light)
        new_line = '|'.join(line)
        input_data.append(new_line)

    return input_data

def build_inputs(time, num_hours_input, num_minutes_input, num_light_input, light):
    input_hour = []
    for i in range(0, num_hours_input):
        input_hour.append(time.hour)
    hour = np.array(input_hour)

    input_minute = []
    for i in range(0, num_minutes_input):
        input_minute.append(time.minute)
    minute = np.array(input_minute)

    input_light = []
    for i in range(0, num_light_input):
        input_light.append(light)
    light_ar = np.array(input_light)

    return hour, minute, light_ar

def calc_week(time):
    if time.weekday() > 4:
        weekday = 1
    else:
        weekday = 0
    return weekday

def plot_hist(Time, Light, PIR_OnOff, State_Trans, Reward, Perf, SC_Volt, PIR, episode, tot_rew, event_detect, tot_events):

    print("Total reward: ", tot_rew)
    #print(Time)
    percent = round(((event_detect / tot_events) * 100), 2)
    #Start Plotting
    plt.figure(1)
    ax1 = plt.subplot(611)
    #plt.title(('Transmitting every {0} sec, PIR {1} ({2} events). Tot reward: {3}').format('60', using_PIR, PIR_events, tot_rew))
    plt.title(('Sim Results: Events Detected: {1} / {2} ({3}%)').format(int(tot_rew), event_detect, tot_events, percent))
    plt.plot(Time, Light, 'b-', label = 'Light', markersize = 10)
    plt.ylabel('Light\n[lux]', fontsize=12)
    plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    ax1.set_xticklabels([])
    plt.grid(True)
    ax2 = plt.subplot(612)
    plt.plot(Time, PIR, 'k.', label = 'PIR detection', markersize = 15)
    plt.ylabel('PIR\nEvents\n[num]', fontsize=12)
    plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc=9, prop={'size': 10})
    ax2.set_xticklabels([])
    plt.grid(True)
    ax3 = plt.subplot(613)
    plt.plot(Time, SC_Volt, 'r.', label = 'SC Voltage', markersize = 15)
    plt.ylabel('SC\nVolt [V]', fontsize=12)
    plt.legend(loc=9, prop={'size': 10})
    plt.ylim(2.2,5.6)
    ax3.set_xticklabels([])
    plt.grid(True)
    ax4 = plt.subplot(614)
    plt.plot(Time, PIR_OnOff, 'y.', label = 'PIR_OnOff', markersize = 15)
    plt.ylabel('PIR\nAction\n[num]', fontsize=12)
    plt.xlabel('Time [h]', fontsize=20)
    plt.legend(loc=9, prop={'size': 10})
    plt.ylim(0)
    ax4.set_xticklabels([])
    plt.grid(True)
    ax5 = plt.subplot(615)
    plt.plot(Time, State_Trans, 'g.', label = 'State Transition', markersize = 15)
    plt.ylabel('State\nTrans\n[num]', fontsize=12)
    plt.xlabel('Time [h]', fontsize=15)
    plt.legend(loc=9, prop={'size': 10})
    plt.ylim(0)
    ax5.set_xticklabels([])
    plt.grid(True)
    ax6 = plt.subplot(616)
    plt.plot(Time, Reward, 'b.', label = 'Reward', markersize = 15)
    plt.ylabel('Reward\n[num]', fontsize=12)
    plt.xlabel('Time [hh:mm]', fontsize=15)
    plt.legend(loc=9, prop={'size': 10})
    #plt.ylim(0)
    xfmt = mdates.DateFormatter('%H:%M')
    ax6.xaxis.set_major_formatter(xfmt)
    plt.grid(True)
    plt.savefig('Graph.png', bbox_inches='tight')
    plt.show()
