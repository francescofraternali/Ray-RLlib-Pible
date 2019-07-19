import glob
import subprocess
import ray
from ray.rllib.agents import ddpg
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
#import arrow

#from sample import *

FONTSIZE = 16
#MAX_AIRFLOW = 1
#MIN_AIRFLOW = 0

#start_time = arrow.get(2018, 4, 1, tzinfo=PT)
#end_time = arrow.get(2018, 4, 5, tzinfo=PT)

# Read data
#observations, actions, consumptions, time_range, scalers = read_data(start_time, end_time, '2146', normalize=True)

# Init exp
from gym_envs.envs.Pible_env import pible_env_creator

register_env('Pible-v2', pible_env_creator)

# Initialize action/obs
pre_action = 0
pre_reward = 0

#assert len(observations.shape) == 2

ray.init()
# lamdb not useful any more
lambds = [1]
learned = []
#iterations = range(5, 501, 500)
rewards_list = []
#for iteration in iterations:
iteration = 5
if True:
    path = glob.glob(subprocess.getoutput('eval echo "~$USER"') + '/ray_results/PPO/PPO_Pible-v2_0_lr=0.01_2019-07-18_17-32-05wunggtds/' +
    #path = glob.glob(subprocess.getoutput('eval echo "~$USER"') + '/ray_results/DDPG/DDPG_VAV-v0_0_2019-05-10_20-02-38zocesjrb' +
                     '/checkpoint_' + str(iteration) + '/checkpoint-' + str(iteration),
                     recursive=True)
    assert len(path) == 1, path
    #agent = ddpg.DDPGAgent(config={
    #    'env_config': {
    #        'max_airflow': 1,
    #        'min_airflow': 0,
    #    }, "input_evaluation": []}, env='VAV-v0')
    agent = ppo.PPOAgent(config=None, env='Pible-v2')
    agent.restore(path[0])
    learned_actions = []
    rewards = []
    for i in range(1,10):
        #   obs, action = observations.iloc[i], actions.iloc[i:i+1].to_list()
        #if True:
        # Note: I change the reward calculation into this way
        #reward = get_reward(consumptions.iloc[i + 1], observations.iloc[i + 1], action)
        reward = 0
        if i == 3:
            test = 1
        else:
            test = 0
        learned_action = agent.compute_action( 
            #observation=[0],
	    observation=[test],
            prev_action=1,
            prev_reward=1
        )
        pre_reward = reward
        pre_action = 0
        print("action learned", learned_action)
        
        #learned_actions.append(learned_action[0])
        #rewards.append(reward)
    quit()
    curr_min = np.min(learned_actions)
    curr_max = np.max(learned_actions)
    print('===================')
    print('iter: {0}'.format(iteration))
    print('min action: {0}'.format(curr_min))
    print('max action: {0}'.format(curr_max))
    #learned_actions = [(act - curr_min) * (MAX_AIRFLOW- MIN_AIRFLOW) / (curr_max - curr_min)  - MIN_AIRFLOW for act in learned_actions]
    learned.append(learned_actions)
    rewards_list.append(rewards)


quit()
#fig = plt.figure(figsize=(24, 16))
#ax = plt.gca()
fig, axes = plt.subplots(3,1)
fig.set_size_inches((24,32))
ax = axes[0]
scaled_ax = axes[1]
ax.grid(True)
ax.tick_params(axis='both', which='major', labelsize=7)
ax.set_title(room + " in CSE Building", fontsize=20)

index = time_range[:-1]
ax.plot_date(index,
             scalers['safsp'].inverse_transform(np.array(actions[:-1]).reshape((-1,1))),
             '-',
             linewidth=1,
             label='Actual Supply Air Flow',
             )
scaled_ax.plot_date(index,
                    actions[:-1],
                    '-',
                    linewidth=1,
                    label='Actual Supply Air Flow',
                    )
ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)

# ax.set_ylabel('Learned Supply Air Flow', fontsize=FONTSIZE)
for i, iteration in enumerate(iterations):
    recovered = scalers['safsp'].inverse_transform(np.array(learned[i]).reshape((-1,1)))
    recovered = recovered.reshape((-1,))
    ax.plot_date(index, recovered, '-', linewidth=1, label=iteration)
    scaled_ax.plot_date(index, learned[i], '-', linewidth=1, label=iteration)
    #ax.plot_date(index, learned[i], '-', linewidth=1, label=iteration)
    axes[2].plot_date(index, rewards_list[i], '-', linewidth=1, label=iteration)

# ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%H'))

ax.legend(fontsize=FONTSIZE)

ax.set_xlabel('day-hour', fontsize=FONTSIZE)
fig.savefig('figs/manual inspect.png', format='png', dpi=500, bbox_inches='tight')
