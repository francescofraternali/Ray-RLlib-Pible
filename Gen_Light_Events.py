"""
Generating Light and Event File from Raw Data
"""

import os
import numpy as np
import datetime
import time
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import sys
import random

#print("Starting Adapting Light...")
#start_sim_1 = sys.argv[1]
#start_sim_2 = sys.argv[2]
#end_sim_1 = sys.argv[3]
#end_sim_2 = sys.argv[4]
#Text = sys.argv[5]
#Input_Data = sys.argv[6]
rand_time = random.randrange(-15, 15, 1)
rand_light = random.randrange(-30, 30, 1)
#print(rand_time, rand_light)
rand_time = 0
rand_light = 0
days = 70
#Text = 'FF66_2150_Middle_Event_RL'
#Text = 'FF21_2146_Corridor_Event_RL'
Text = 'FF5_2106_Door_Event_Battery'

try:
    os.remove(Text + "_Adapted.txt")
except:
    pass

with open(Text + ".txt") as f:
    content = f.readlines()
#content = [x.strip() for x in content] # you may also want to remove whitespace characters like `\n` at the end of each line
leng = len(content)-1

# Let's first find the last valid date
for i in range(0, len(content)):
    line = content[leng].split('|')
    if len(line) > 7:
        end_time = datetime.datetime.strptime(line[0], '%m/%d/%y %H:%M:%S') + datetime.timedelta(minutes=rand_time)
        starting_time = end_time - datetime.timedelta(days)
        break
    else:
        leng -= 1

#print(starting_time, end_time)
for i in range(0, len(content)):
    line = content[i].split("|")
    if len(line) > 7:
        curr_time = datetime.datetime.strptime(line[0], '%m/%d/%y %H:%M:%S')
        if curr_time < starting_time:
            pass  # don't do anything since this date is too early to be considered
        else:
            sc_volt = int(line[5])
            Text_split = Text.split('_')
            light = int(line[8])
            new_light = int(light + (light/100) * rand_light)
            #print(light, new_light)
            if sc_volt is not 0 or 'Battery' in Text_split:
                curr_time = curr_time + datetime.timedelta(minutes=rand_time)
                curr_time_new = curr_time.strftime('%m/%d/%y %H:%M:%S')
                line[0] = curr_time_new
                line[8] = str(new_light)
                new_line = '|'.join(line)
                #print(content[i], new_line)
                #exit()
                with open(Text + '_Adapted.txt', "a") as myfile:
                    myfile.write(new_line)

print("Done with: " + Text + "_Adapted.txt")
