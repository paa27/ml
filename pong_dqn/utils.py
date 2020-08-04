'''
Utilities for training agents
'''

import matplotlib.pyplot as plt
import torch

def train_agent(agent, env, episodes, echo = True, echo_freq = 10, save_frames = False, save_metrics = False):
    
    try:
        
        metrics = {'reward':[]}
        frames = []

        for episode in range(episodes):
            observation = env.reset()
            done = False
            agent.history.append(observation)


            while not done:
                action = agent.act()
                observation, done, reward, info = env.step(action)
                next_action = agent.act()
                agent.memorize(action, reward, observation, done)
                agent.learn()

                if save_metrics:
                    metrics['reward'].append(reward)
                if save_frames:
                    frames.append(env.render(mode='rgb_array'))

            if echo and (episode % echo_freq == 0):
                print('Episode {} completed'.format(episode))
                
    except (KeyboardInterrupt, SystemExit):
        torch.save(agent.state_dict(), 'agent_except.state')
            
    return metrics,frames
    
    
def render_frames(frames):
    
    img = plt.imshow(frames[0]) # only call this once
    for frame in frames[1:]:
        img.set_data(frame) # just update the data
        display.display(plt.gcf())
        display.clear_output(wait=True)