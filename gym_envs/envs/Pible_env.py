"""
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from time import sleep
import datetime
import subprocess
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os


# Light_Path
path = subprocess.getoutput('eval echo "~$USER"') + "/Desktop/Ray-RLlib-Pible/gym_envs/envs"

# using PIR = 1 if you want to include the PIR in the simulation
state_trans = 900 # time in seconds
using_PIR = 0
PIR_events = 200 # Number of PIR events detected during a day. This could happen also when light is not on

I_PIR_detect = 0.000102; PIR_detect_time = 2.5

# Super-Capacitor values
SC_volt_min = 2.3; SC_volt_max = 5.4; SC_size = 1.5; SC_begin = 4.0
SC_volt_die = 3.0
# Solar Panel values
V_solar_200lux = 1.5; I_solar_200lux = 0.000031;

# Communication values
I_Wake_up_Advertise = 0.00006; Time_Wake_up_Advertise = 11
I_BLE_Comm = 0.00025; Time_BLE_Comm = 4
I_BLE_Sens_1= ((I_Wake_up_Advertise * Time_Wake_up_Advertise) + (I_BLE_Comm * Time_BLE_Comm))/(Time_Wake_up_Advertise + Time_BLE_Comm)
Time_BLE_Sens_1 = Time_Wake_up_Advertise + Time_BLE_Comm

if using_PIR == 1:
	I_sleep = 0.0000055
else:
	I_sleep = 0.0000032

class PibleEnv(gym.Env):
    """
    Description:
	A representation o the Pible-mote as a gym environment to test RL algorithms
    Observation: 
        Type: Box(4)
        Num	Observation                 		Min     	Max
        0	Super-Capacitor Voltage level		2.3		5.4
	#0	Cart Position             		-4.8            4.8
        #1	Cart Velocity             		-Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	15 min sensing-rate
	1	15 sec sensing-rate
        
    Reward:
        Reward equal to action if node alive (i.e. super capcitor voltage level > 2.3) else very negative (i.e. -300)
    Starting State:
        The super-capacitor voltage level is 4 volts
    Episode Termination:
        Episode terminates after 24 hours of simulation
    """
    
    #metadata = {
    #    'render.modes': ['human', 'rgb_array'],
    #    'video.frames_per_second' : 50
    #}

    def __init__(self, config):
        self.count = 0
        self.light_count = 0
        self.done = 0

        with open(path + "/Light_sample.txt", 'r') as f:
            content = f.readlines()
        self.light_input = [x.strip() for x in content]

        # Building the observation
        high = np.array([SC_begin])
            #self.state,
            #np.finfo(np.float32).max,
            #self.theta_threshold_radians * 2,
            #np.finfo(np.float32).max
	#])
        min_sc = np.array([SC_volt_min])
        max_sc = np.array([SC_volt_max])
        min_act = np.array([15]) # seconds
        max_act = np.array([900]) # seconds

        self.action_space = spaces.Discrete(4)
        #self.action_space = spaces.Box(min_act, max_act, dtype=np.float32)
        self.observation_space = spaces.Box(min_sc, max_sc, dtype=np.float32)

    def step(self, action):

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #action += self.min_act 

	# Reading next light value
        line = self.light_input[self.light_count]
        self.light_count += 1
        line = line.split('|')
        self.time = self.time + datetime.timedelta(0, state_trans)
        self.PIR = int(line[7])
        light_pure = int(line[8])

        if self.time >= self.end_time:
            self.done = 1
        if self.light_count == len(self.light_input)-1:
            self.light_count = 0
            self.done = 1

        self.reward = reward_func(action, self.SC_volt)
        self.SC_volt = energy_calc(self.SC_volt, light_pure, action, self.PIR)
    
        self.SC_Volt.append(self.SC_volt)
        self.Reward.append(self.reward)
        self.PIR_hist.append(self.PIR)
        #self.Perf.append(self.perf)
        self.Time.append(self.time)
        self.Light.append(light_pure)
        self.Action.append(action)
    
        return np.array([self.SC_volt]), self.reward, self.done, {}

    def reset(self):
        self.SC_Volt = []; self.SC_Norm = []; self.Reward = []; self.PIR_hist = []; self.Perf = []; self.Time = []; self.Light = []; self.Action = []; self.light_count == 0
        
        with open(path + "/Light_sample.txt", 'r') as f:
            for line in f:
                line = line.split('|')
                self.time = datetime.datetime.strptime(line[self.light_count], '%m/%d/%y %H:%M:%S')
                break

        self.end_time = self.time + datetime.timedelta(0, 24*60*60)
        self.done = 0
        self.PIR = 0
        self.SC_volt = SC_begin

        return np.array([self.SC_volt])


    def render(self, episode, tot_rew):
        plot_hist(self.Time, self.Light, self.Action, self.Reward, self.Perf, self.SC_Volt, self.SC_Norm, self.PIR_hist, episode, tot_rew)


def reward_func(action, SC_volt):

    reward = action
    #reward = int(state_trans/action)
    
    
    if SC_volt <= SC_volt_die:
        reward = -300
    if action == 900.0 and SC_volt <= SC_volt_die:
        reward = -100
    '''
    if perf == 300.0:
        reward = 1
        print("trovato")
    else:
        reward = 0
    '''
    return reward

def energy_calc(SC_volt, light, action, PIR):

    try: 
        SC_volt = SC_volt[0]
    except:
        pass
    
    #effect = state_trans/action
    if action == 3:
        effect = 60
    elif action == 2:
        effect = 15
    elif action == 1:
        effect = 3
    else:
        effect = 1

    Energy_Rem = SC_volt * SC_volt * 0.5 * SC_size

    if SC_volt <= SC_volt_min: # Node is death and not consuming energy
        #Energy_Prod = state_trans * V_solar_200lux * I_solar_200lux * (light/200)
        Energy_Used = 0
    else: # Node is alive
        Energy_Used = ((state_trans - (Time_BLE_Sens_1 * effect)) * SC_volt * I_sleep) # Energy Consumed by the node in sleep mode 
        Energy_Used += (Time_BLE_Sens_1 * effect * SC_volt * I_BLE_Sens_1) # energy consumed by the node to send data
        #Energy_Used += (PIR * I_PIR_detect * PIR_detect_time) # energy consumed by the node for the PIR

    Energy_Prod = state_trans * V_solar_200lux * I_solar_200lux * (light/200)

    # Energy cannot be lower than 0
    Energy_Rem = max(Energy_Rem - Energy_Used + Energy_Prod, 0)

    SC_volt = np.sqrt((2*Energy_Rem)/SC_size)

    # Setting Boundaries for Voltage
    if SC_volt > SC_volt_max:
        SC_volt = SC_volt_max

    if SC_volt < SC_volt_min:
        SC_volt = SC_volt_min

    #SC_volt = np.round(SC_volt, 4)

    try: 
        SC_volt = SC_volt[0]
    except:
        pass
    
    return SC_volt

def plot_hist(Time, Light, Action, Reward, Perf, SC_Volt, SC_Norm, PIR, episode, tot_rew):

    #Start Plotting
    plt.figure(1)
    plt.subplot(411)
    plt.title(('Simulation while sensing every {0} sec and PIR {1} with {2} events').format(state_trans, using_PIR, PIR_events))
    plt.plot(Time, Light, 'b-', label = 'SC Percentage', markersize = 10)
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
    plt.show()

    '''
    #Start Plotting
    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    plt.plot(Time, Light, 'b', label = 'Light')
    plt.plot(Time, Action, 'y*', label = 'Action',  markersize = 15)
    plt.plot(Time, Reward, 'k+', label = 'Reward')
    #plt.plot(Time, Perf, 'g', label = 'Performance')
    plt.plot(Time, SC_Volt, 'r+', label = 'SC_Voltage')
    #plt.plot(Time, SC_Norm, 'm^', label = 'SC_Voltage_Normalized')
    plt.plot(Time, PIR, 'c^', label = 'Occupancy')
    xfmt = mdates.DateFormatter('%m-%d-%y %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.tick_params(axis='both', which='major', labelsize=10)
    legend = ax.legend(loc='center right', shadow=True)
    plt.legend(loc=9, prop={'size': 10})
    plt.title('Epis: ' + str(episode) + ' tot_rew: ' + str(tot_rew), fontsize=15)
    plt.ylabel('Super Capacitor Voltage[V]', fontsize=15)
    plt.xlabel('Time[h]', fontsize=20)
    ax.grid(True)
    #fig.savefig('Saved_Data/Graph_hist_' + Text + '.png', bbox_inches='tight')
    plt.show()
    #plt.close(fig)
    '''

def pible_env_creator(env_config):
    return PibleEnv(env_config)

