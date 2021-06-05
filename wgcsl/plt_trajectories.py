
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from numpy.core.fromnumeric import size
from numpy.ma.core import right_shift
import seaborn as sns; sns.set()
import glob2
import pickle


def load_buffer(path):
    with open(path, "rb") as fp:  
        buffer = pickle.load(fp)  
    return buffer

def plot_trajectories(buffer, name=''):
    ags = buffer['ag']
    gs = buffer['g']
    num_episode = gs.shape[0]
    for i in range(num_episode):
        plt.figure(figsize=(5,5))
        dx = ags[i][1:, 0] - ags[i][:-1, 0]
        dy = ags[i][1:, 1] - ags[i][:-1, 1]

        # plt.quiver(ags[i][:-1, 0], ags[i][:-1, 1], dx, dy, angles='xy', scale=1, scale_units='xy',headlength=4, headwidth=3, color='darkgray') #
        plt.scatter(ags[i][0][0], ags[i][0][1], marker='^', c='g', s=150, label='Starting Position')
        # plt.scatter(ags[i][:, 0], ags[i][:, 1], marker='.', s=50, label='Achieved Goals')
        plt.plot(ags[i][:, 0], ags[i][:, 1], marker='.', label='Achieved Goals', alpha=0.7, linewidth=3)
        plt.scatter(gs[i][0][0], gs[i][0][1], marker='*', c='r', s=160, label='Desired Goal')
        plt.legend()
        if 'Point' in name:
            plt.xlim(-5,5)
            plt.ylim(-5,5)
        elif 'Reacher' in name:
            plt.xlim(-0.25,0.25)
            plt.ylim(-0.25,0.25)
        plt.savefig(name + 'trajectory_' + str(i) + '.pdf')
    
def plot_trajectories_selected(buffer, name, select_ids):
    colors = ['b', 'orange', 'g', 'r', 'gold']
    #['skyblue', 'orange', 'lightgreen', 'gold', 'red']
    ags = buffer['ag']
    gs = buffer['g']
    num_episode = gs.shape[0]
    plt.figure(figsize=(5,5))
    idx = 0
    for i in select_ids:
        dx = ags[i][1:, 0] - ags[i][:-1, 0]
        dy = ags[i][1:, 1] - ags[i][:-1, 1]

        # plt.quiver(ags[i][:-1, 0], ags[i][:-1, 1], dx, dy, angles='xy', scale=1, scale_units='xy',headlength=4, headwidth=3, color='darkgray') #
        plt.scatter(ags[i][0][0], ags[i][0][1], marker='^', c=colors[idx], s=120, label='Starting Position')
        # plt.scatter(ags[i][:, 0], ags[i][:, 1], marker='.', s=50, label='Achieved Goals')
        plt.plot(ags[i][:, 0], ags[i][:, 1], marker='.', c=colors[idx], label='Achieved Goals', alpha=0.45, linewidth=3)
        plt.scatter(gs[i][0][0], gs[i][0][1], marker='*', c=colors[idx], s=150, label='Desired Goal')
        idx += 1
    plt.legend(['Starting Position', 'Achieved Goals', 'Desired Goal'])
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.savefig(name + 'trajectory_selected' + '.png')
    

if __name__ == '__main__':
    envs = ['Point2DLargeEnv-v1', 'Point2D-FourRoom-v1', 'SawyerReachXYZEnv-v1', 'FetchReach-v1', 'Reacher-v2', 'SawyerDoor-v0']  
    path = '/Users/yangrui/Desktop/AWGCSL/awgcsl/data/evaluate_gcsl/'
    env_id = 4
    path += envs[env_id] + '/buffer.pkl'
    buffer = load_buffer(path)
    save_path = './data/gcsl_trajs_' + envs[env_id] + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plot_trajectories(buffer, save_path)
    # plot_trajectories_selected(buffer, save_path, select_ids=[6,12,11])   # 2,3,
    # [1,2,4,11,16]