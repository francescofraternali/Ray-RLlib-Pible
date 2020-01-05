import datetime
import json
import numpy


from scipy.spatial.distance import jensenshannon
from numpy import asarray

path_light = "FF66_2150_Middle_Event_RL_Adapted.txt"
light_divider = 2
Prob_list = []

def PIR_pdf(start_date, end_date):
    light_list = []
    file_data = []
    PIR_list = []

    start_date = datetime.datetime.strptime(start, '%m/%d/%y %H:%M:%S')
    end_date = datetime.datetime.strptime(end, '%m/%d/%y %H:%M:%S')
    end_date_temp = start_date + datetime.timedelta(hours=1)
    path_light_data = path_light
    PIR = 0
    tot_events = 0
    with open(path_light_data, 'r') as f:
        for line in f:
            line_split = line.split("|")
            checker = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')
            if start_date <= checker and checker <= end_date_temp:
                file_data.append(line)
                light_list.append(int(line_split[8]))
                PIR += int(line_split[6])
            if checker > end_date_temp:
                start_date = end_date_temp
                end_date_temp = start_date + datetime.timedelta(hours=1)
                PIR_list.append(PIR)
                tot_events += PIR
                PIR = 0
            if checker >= end_date:
                PIR_list_day = [i/tot_events for i in PIR_list]
                break
    print(PIR_list_day)
    return PIR_list_day

for i in range(0,1):
    start = "12/02/19 00:00:00"
    end = "12/03/19 00:00:00"
    Prob_list.append(PIR_pdf(start, end))
    start = "12/04/19 00:00:00"
    end = "12/05/19 00:00:00"
    Prob_list.append(PIR_pdf(start, end))
    start = "12/05/19 00:00:00"
    end = "12/06/19 00:00:00"
    Prob_list.append(PIR_pdf(start, end))
    start = "12/06/19 00:00:00"
    end = "12/07/19 00:00:00"
    Prob_list.append(PIR_pdf(start, end))
    start = "12/07/19 00:00:00"
    end = "12/08/19 00:00:00"
    Prob_list.append(PIR_pdf(start, end))




p = asarray(Prob_list[0])
q = asarray(Prob_list[1])
# calculate JS(P || Q)
js_pq = jensenshannon(p, q, base=2)
print('JS(P || Q) Distance: %.3f' % js_pq)

p = asarray(Prob_list[1])
q = asarray(Prob_list[2])
# calculate JS(P || Q)
js_pq = jensenshannon(p, q, base=2)
print('JS(P || Q) Distance: %.3f' % js_pq)

p = asarray(Prob_list[2])
q = asarray(Prob_list[3])
# calculate JS(P || Q)
js_pq = jensenshannon(p, q, base=2)
print('JS(P || Q) Distance: %.3f' % js_pq)

p = asarray(Prob_list[3])
q = asarray(Prob_list[4])
# calculate JS(P || Q)
js_pq = jensenshannon(p, q, base=2)
print('JS(P || Q) Distance: %.3f' % js_pq)

exit()


bins = numpy.linspace(0, 1000, 10)
#print(bins)

data = light_list
#print(light_list)
#print(sum(light_list)/len(light_list))
digitized = numpy.digitize(data, bins)
#print("digit", digitized)
lenght = len(light_list)
#bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
#print(bin_means)
#print(numpy.histogram(data, bins, weights=data)[0])
#print(numpy.histogram(data, bins)[0])
bin_means = (numpy.histogram(data, bins, weights=data)[0] /
             numpy.histogram(data, bins)[0])
day_1 = numpy.histogram(data, bins)[0]/lenght
day_2 = numpy.histogram(data, bins)[0]/lenght
print(day_1)

# calculate the jensen-shannon distance metric
# define distributions
p = asarray(day_1)
q = asarray(day_2)
# calculate JS(P || Q)
js_pq = jensenshannon(p, q, base=2)
print('JS(P || Q) Distance: %.3f' % js_pq)
# calculate JS(Q || P)
js_qp = jensenshannon(q, p, base=2)
#print('JS(Q || P) Distance: %.3f' % js_qp)
