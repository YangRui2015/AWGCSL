
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from numpy.core.fromnumeric import size
from numpy.ma.core import right_shift
import seaborn as sns; sns.set()
import glob2
import argparse

smooth = True

def smooth_reward_curve(x, y):
    # halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution÷
    halfwidth = 2
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    try:
        data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    except:
        import pdb; pdb.set_trace()
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


# Load all data.
def load_data(dir, key='test/success_rate', filename='progress.csv', x_time='epoch'):
    data = []
    # find all */progress.csv under dir
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(dir, '**', filename))]
    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        results = load_results(os.path.join(curr_path, filename))
        if not results:
            print('skipping {}'.format(curr_path))
            continue
        print('loading {} ({})'.format(curr_path, len(results['epoch'])))

        success_rate = np.array(results[key])  #[:50]
        epoch = np.array(results['epoch']) + 1  #[:50]
        episodes = np.array(results['train/episode']) - 20
        # Process and smooth data.
        assert success_rate.shape == epoch.shape
        if x_time == 'epoch':
            x = epoch
        elif x_time == 'episode':
            x = episodes
        else:
            print('No such x axis label')
            import pdb;pdb.set_trace()
        y = success_rate
        if smooth:
            x, y = smooth_reward_curve(x, y)
        assert x.shape == y.shape
        data.append((x, y))
    return data

def load_datas(dirs, key='test/success_rate', filename='progress.csv', x_time='epoch'):
    datas = []
    for dir in dirs:
        data = load_data(dir, key, filename, x_time)
        datas.append(data)
    return datas

# Plot datas
def plot_datas(datas, labels, info, fontsize=16, i=0, j=0, method='median'):
    # plt.clf()
    title, xlabel, ylabel = info
    for data, label in zip(datas, labels):
        try:
            xs, ys = zip(*data)
        except:
            import pdb; pdb.set_trace()
        xs, ys = pad(xs), pad(ys)
        assert xs.shape == ys.shape
        if method == 'median':
            plt.plot(xs[0], np.nanmedian(ys, axis=0), label=label)
            plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25)
        elif method == 'mean':
            y_mean, y_std = np.mean(ys, axis=0), np.std(ys, axis=0)
            plt.plot(xs[0], y_mean, label=label)
            plt.fill_between(xs[0], y_mean - y_std, y_mean + y_std, alpha=0.25)
        else:
            print('no such plot method')
            import pdb;pdb.set_trace()

    
    plt.title(title, fontsize=fontsize-4)
    plt.xlabel(xlabel, fontsize=fontsize)
    if j == 0:
        plt.ylabel(ylabel, fontsize=fontsize)
    if j == 0:
        plt.legend(fontsize=fontsize-6) #, loc=4, bbox_to_anchor=(0.5, 0.06, 0.5, 0.5))
    plt.xticks(fontsize=fontsize-4)
    plt.yticks(fontsize=fontsize-4)

def plot_main(dirs, labels, info, key='test/success_rate', filename='progress.csv', save_dir='./test.png', x_time='epoch'):
    plt.figure(dpi=300, figsize=(5,4))
    sns.set(style='whitegrid')
    datas = load_datas(dirs, key, filename, x_time)

    plot_datas(datas, labels, info)
    plt.subplots_adjust(left=0.15, right=0.98, bottom=0.14, top=0.92, hspace=0.3, wspace=0.15)
    plt.savefig(save_dir)

def subplot_main(pic_dirs, labels, infos, row=1, key='test/success_rate',\
                    filename='progress.csv', save_dir='./test.png', x_time='epoch'):
    n = len(pic_dirs)  # number of pics
    col = math.ceil(n / row)
    fig, axes = plt.subplots(row, col, figsize=(4.25 * col,4))
    fig.subplots_adjust(left=0.035, right=0.99, bottom=0.155, top=0.92, hspace=0.2, wspace=0.14)
    sns.set(style='white')
    titles, x_label, y_label = infos
    xy_label = [x_label, y_label]
    for i in range(row):
        for j in range(col):
            idx = i * col + j 
            if idx + 1 > n:
                break
            
            plt.subplot(row, col, idx+1)
            datas = load_datas(pic_dirs[idx], key, filename, x_time)
            plot_datas(datas, labels, [titles[idx], *xy_label], i=i, j=j)

    plt.savefig(save_dir)

if __name__ == "__main__":
    key_rate = 'test/success_rate'
    # key_Q = 'test/mean_Q'  #Eval success ratio

    # one figure
    # save_dir = '/Users/yangrui/Desktop/Model-basedHER-main/pics/test.png'
    # prefix_dir = '/Users/yangrui/Desktop/Model-basedHER-main/data/'
    
    # labels = ['GCSL', 'GCSL+gamma',  'GCSL+gamma+exp_adv+pos', 'GCSL+gamma+tanh_adv+pos', 'GCSL+tanh_adv+pos', 'GCSL+gamma+01_adv+pos'] # 'GCSL+gamma+exp_adv+clip-', 'GCSL+gamma+01_adv+pos',
    # environments = ['Point2DLargeEnv-v1', 'Point2D-FourRoom-v1', 'Reacher-v2', 'FetchReach-v1', 'SawyerReachXYZEnv-v1']
    # sub_dir = environments[3]
    # title = sub_dir
    # info = [title, 'Epoch', 'Median Success Rate']
    # # info = [title, 'Epoch', 'Mean Q Value']
    # dirs = ['gcsl_new/' +sub_dir, 'gamma_gcsl_new/' + sub_dir,  \
    #         'gamma_gcsl_exp_adv_pos_noclip/' + sub_dir, 
    #         'gamma_gcsl_adv_pos_tanh/' + sub_dir, 'gcsl_adv_pos_tanh/' + sub_dir, 'gamma_gcsl_adv_pos_01/' + sub_dir] # 'gamma_gcsl_adv/' + sub_dir,
    # dirs = [os.path.join(prefix_dir, x) for x in dirs]
    # plot_main(dirs, labels, info, key=key_rate, save_dir=save_dir)
    # import pdb; pdb.set_trace()

    ###############################
   


    ##################################
    # # # # subplot figures
    save_dir = './test_6.png'
    prefix_dir = '/Users/yangrui/Desktop/logs_weighted/latest/'
    
    # temp = 'gamma_exp_adv_gcsl/'
    # path_list = ['', 'no_target_new_param/', 'ordered_buffer_5e4/','small_buffer1e4/', 'small_buffer2e3/']
    # path_list = [x + temp for x in path_list]
    path_list = ['gcsl/', 'gamma_gcsl/', 'gamma_tanh_adv_gcsl/', 'gamma_exp_adv_gcsl/', 'gamma_adv_gcsl/',
                'tanh_adv_gcsl/', 'exp_adv_gcsl/', 'adv_gcsl/', 'gamma_exp_adv_clip10_gcsl/']  
    pic_title = ['Point2DLargeEnv-v1', 'Point2D-FourRoom-v1', 'FetchReach-v1', 'SawyerReachXYZEnv-v1', 'Reacher-v2', 'SawyerDoor-v0']  
  
    pic_dirs = [[[] for _ in range(len(path_list))] for _ in range(len(pic_title))]
    for i in range(len(pic_title)):
        for j in range(len(path_list)):
            pic_dirs[i][j] = prefix_dir + path_list[j] + '/' + pic_title[i]
 
    infos = [pic_title, 'Episodes', 'Average Return'] #'Median Success Rate']
    row = 1
    # legend = ['origin', 'new param/no target', 'ordered buffer 5e4','ordered buffer 1e4', 'ordered buffer 2e3']
    legend = ['GCSL', 'GCSL+gamma', 'GCSL+gamma+tanh_adv', 'GCSL+gamma+exp_adv','gamma+adv', 'tanh_adv', 'exp_adv', 'adv', 'GCSL+gamma+exp_adv clip10'] 
    subplot_main(pic_dirs, legend, infos, row, key='test/return', save_dir=save_dir, x_time='episode')




