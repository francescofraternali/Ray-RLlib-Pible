"""
Pible Simulator for openAI Gym
"""

import tensorflow as tf
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


# Parameters to change
state_trans = 900 # State transition in seconds. I.e. the system takes a new action every "state_trans" seconds. It also corresponds to the communication time
sens_time = 60 # Sensing time in seconds. 
transm_thres_light = 100 # value in lux
SC_begin = 4.0 # Super Capacitor initial voltage level. Put a number between 5.4 (i.e. max) and 2.3 (i.e. min) 
SC_volt_die = 3.0 # Voltage at which the simulator consider the node death
using_temp = False
using_light = True
using_PIR = True # PIR active and used to detect people
PIR_events = 100 # Number of PIR events detected during a day. This could happen also when light is not on
using_Accelerometer = False # Activate only if using Accelerometer

# DO NOT MODIFY! POWER CONSUMPTION PARAMETERS! Change them only if you change components.
SC_volt_min = 2.3; SC_volt_max = 5.5; SC_size = 1.5; SC_volt_die = 3.0

# Board and Components Consumption
i_sleep = 0.0000032;
i_sens =  0.000100; time_sens = 0.2
i_PIR_detect = 0.000102; time_PIR_detect = 2.5
i_accel_sens = 0.0026; accel_sens_time = 0.27

if using_PIR == True:
    i_sleep += 0.000001
    if PIR_events != 0:
        PIR_events_time = (24*60*60)/PIR_events  # PIR events every "PIR_events_time" seconds. Averaged in a day
    else:
        PIR_events_time = 0
else:
    PIR_events = 0

# if using_Accelerometer:
#    i_sleep += 0.000008

# Communication (wake up and transmission) and Sensing Consumption
i_wake_up_advertise = 0.00006; time_wake_up_advertise = 11
i_BLE_comm = 0.00025; time_BLE_comm = 4
i_BLE_sens = ((i_wake_up_advertise * time_wake_up_advertise) + (i_BLE_comm * time_BLE_comm))/(time_wake_up_advertise + time_BLE_comm)
time_BLE_sens = time_wake_up_advertise + time_BLE_comm

#i_BLE_sens = 0.000210; time_BLE_sens = 6.5

# Solar Panel Production
v_solar_200_lux = 1.5; i_solar_200_lux = 0.000031
p_solar_1_lux = (v_solar_200_lux * i_solar_200_lux) / 200.0

# Light_Path
#path = subprocess.getoutput('eval echo "~$USER"') + "/Desktop/Ray-RLlib-Pible/gym_envs/envs"

min_act = np.array([0.0]) # min action 
max_act = np.array([3.0]) # max action 
max_sens = 15.0 # minimum sensing rate in seconds
min_sens = 900.0 # maximum sensing rate in seconds

l = [sens_time, PIR_events_time, state_trans]

class PibleEnv(gym.Env):
    """
    Description:
	A representation o the Pible-mote as a gym environment to test RL algorithms
    Observation (not updated): 
        Type: Box(4)
        Num	Observation                 		Min     	Max
        0	Super-Capacitor Voltage level		2.3		5.4
	#0	Cart Position             		-4.8            4.8
        #1	Cart Velocity             		-Inf            Inf
        
    Actions (not updated):
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

    def __init__(self, config):
        self.path = config["path"]
        self.count = 0
        self.light_count = 0
        self.done = 0
        
        with open(self.path + "/Light_sample.txt", 'r') as f:
            content = f.readlines()
        self.light_input = [x.strip() for x in content]

        # Building the observation
        high = np.array([SC_begin])
            #self.state,
            #np.finfo(np.float32).max,
            #self.theta_threshold_radians * 2,
            #np.finfo(np.float32).max
	#])
        min_ = np.array([
            2.1,
            0
            ])
        max_ = np.array([
            5.4,
            24
            ])
        min_sc = np.array([SC_volt_min])
        max_sc = np.array([SC_volt_max])
        min_act = np.array([0]) # min action 
        max_act = np.array([23]) # max action
        #min_sens = 15.0 # minimum sensing rate in seconds
        #max_sens = 900.0 # maximum sensing rate in seconds
        self.action_space = spaces.Discrete(2)
        #self.action_space = spaces.Box(min_act, max_act, dtype=np.float32)
        #self.observation_space = spaces.Box(min_act, max_act, dtype=np.float32)
        self.observation_space = spaces.Discrete(24)

        #self.observation_space = spaces.Tuple((
        #    spaces.Box(
        #        min_sc, max_sc, dtype=np.float32),
        #    spaces.Discrete(24)
        #    #    np.array([0]), np.array([23]))
        #    #0.0, 23.0, shape = (1))
        #))

    def step(self, action):
        #print(action)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #action += self.min_act 

	# Reading next light value
        line = self.light_input[self.light_count]
        self.light_count += 1
        line = line.split('|')
        self.PIR = int(line[7])
        #light_pure = int(line[8])
        light_pure = int(light_env(self.time.hour))
       
        self.SC_volt, time_passed, event = next_event_energy(self.SC_volt, light_pure, self.time.hour)
        
        self.time = self.time + datetime.timedelta(0, time_passed)

        if self.time >= self.end_time:
            self.done = 1
            #print(self.time, self.end_time)
        if self.light_count == len(self.light_input) - 1:
            #print("here", self.light_count)
            self.light_count = 0
            self.done = 1
            
        self.reward = reward_func(action, self.SC_volt, event, self.time.hour)
        #self.SC_volt = energy_calc(self.SC_volt, light_pure, action, self.PIR, event)
    
        self.SC_Volt.append(self.SC_volt)
        self.Reward.append(self.reward)
        self.PIR_hist.append(event)
        self.Time.append(self.time)
        self.Light.append(light_pure)
        self.Action.append(action)
    
        #return np.array([self.time.hour]), self.reward, self.done, {}
        #print(np.array([self.time.hour]))
        #return (np.array([self.SC_volt]), self.time.hour), self.reward, self.done, {}
        return self.time.hour, self.reward, self.done, {}


    def reset(self):
        self.SC_Volt = []; self.SC_Norm = []; self.Reward = []; self.PIR_hist = []; self.Perf = []; self.Time = []; self.Light = []; self.Action = []; self.light_count == 0
        
        with open(self.path + "/Light_sample.txt", 'r') as f:
            for line in f:
                line = line.split('|')
                time_init = datetime.datetime.strptime(line[self.light_count], '%m/%d/%y %H:%M:%S')
                self.time = time_init.replace(hour=0)
                break

        self.end_time = self.time + datetime.timedelta(0, 24*60*60)
        self.done = 0
        self.PIR = 0
        self.SC_volt = SC_begin

        #print('reset')
        #sleep(3)
        #return np.array([2.0])
        return 2
        #return (np.array([self.SC_volt]), 2)


    def render(self, episode, tot_rew):
        plot_hist(self.Time, self.Light, self.Action, self.Reward, self.Perf, self.SC_Volt, self.SC_Norm, self.PIR_hist, episode, tot_rew)


def reward_func(action, SC_volt, event, time):
    
    #reward = action[0]

    #reward = int(state_trans/action)
    if action == 1 and time >= 8 and time <= 10:
        reward = 20
    else:
        reward = -1
    
    if action == 1 and time >=17 and time <= 19:
        reward = 20
        #print("assaggia 0")
    else:
        reward = -1

    if action == 0:
        reward = 1


    if SC_volt <= SC_volt_die:
        reward = -300
    
    # build a list in which you put the time fo the day in which the events happens. The list che be modified based on new events
    
    #if action == 900.0 and SC_volt <= SC_volt_die:
    #    reward = -100

    return reward

def light_env(time):
    #print(time)
    if time > 8 and time < 16:
        light = 250
    else:
        light = 0

    return light

def check_event(time):
    if time == 8:
        event = 1
    else:
        event = 0
    
    return event

def next_event_energy(SC_volt, light, time):

    # next even selection calculation
    orig = [60, 2500, 900]
    dic = {'0': "sens_time", '1' : "PIR_events_time", '2' : "state_trans"}
    x = l.index(min(l)) # consider the earliest event
    time_passed = l[x]

    #print(x, time_passed) 
    for i in range(0, len(l)):
        if i != x:
            l[i] -= time_passed
            #print("hee")
        else:
            l[i] = orig[i]

    #print(l)
    time_passed = 900
    event = check_event(time)    
    #return dic[str(x)], time_passed 

    #def energy_calc(SC_volt, light, event):
    
    # Energy expenditure and remaining calculation

    try: 
        SC_volt = SC_volt[0]
    except:
        pass
    
    #action_norm = (((action - min_act) * (max_sens - min_sens)) / (max_act - min_act)) + min_sens 
    #effect = state_trans/action_norm
    
    #dic_ener = {"sens_time" : 1, "PIR_events_time" : 60, "state_trans" : 10}
    #effect = dic_ener[event]

    Energy_Rem = SC_volt * SC_volt * 0.5 * SC_size

    if SC_volt <= SC_volt_min: # Node is death and not consuming energy
        #Energy_Prod = state_trans * V_solar_200lux * I_solar_200lux * (light/200)
        Energy_Used = 0
    else: # Node is alive
        #if dic[str(x)] == 'sens_time': # Add energy for sensing on energy used
        Energy_Used = (SC_volt * i_sens) * time_sens
        time_sleep = time_passed - time_sens
        if dic[str(x)] == 'PIR_events_time':
            Energy_Used += (SC_volt * i_PIR_detect) * time_PIR_detect
            time_sleep -= time_PIR_detect
        if dic[str(x)] == 'state_trans': # Energy used to send a data
            Energy_Used += (time_BLE_sens * SC_volt * i_BLE_sens) # energy consumed by the node to send data
            time_sleep -= time_BLE_sens
        # finally add time that was sleeping
        Energy_Used = (time_sleep * SC_volt * i_sleep) # Energy Consumed by the node in sleep mode 
        #Energy_Used += (time_BLE_sens * effect * SC_volt * i_BLE_sens) # energy consumed by the node to send data
        #Energy_Used += (PIR * I_PIR_detect * PIR_detect_time) # energy consumed by the node for the PIR

    Energy_Prod = time_passed * p_solar_1_lux * light

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
    
    return SC_volt, time_passed, event



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

def pible_env_creator(env_config):
    return PibleEnv(env_config)

