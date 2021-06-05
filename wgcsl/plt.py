
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
    halfwidth = 3
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

        success_rate = np.array(results[key])  
        epoch = np.array(results['epoch']) + 1 
        if 'her' in dir or 'offline' in dir:
            episodes = np.array(results['train/episode']) 
            if 'offline' in dir and ('Fetch' in dir or 'Reacher' in dir or 'Door' in dir):
                episodes *= 4
        else:
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
        # if 'bcq' in dir:
        #     if 'Reacher' in dir or 'Door' in dir:
        #         x = x[:100]
        #         y = y[:100]
        #     else:
        #         x = x[:40]
        #         y = y[:40]
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

def plot_ppo(env_name, metric, method):
    dic = {'Point2DLargeEnv-v1':'pointlarge',
        'Point2D-FourRoom-v1':'point2d',
        'FetchReach-v1': 'fetch_reach',
        'Reacher-v2':'reacher',
        'SawyerReachXYZEnv-v1':'sawyerxyz',
        'SawyerDoor-v0':'sawyer'}
    key = dic[env_name]
    data = np.load('./results_update0510.npy', allow_pickle=True).item()
    results = data[key]
    if metric == 'test/success_rate':
        y = results[:,:, -1]
    elif metric == 'test/return':
        y = results[:, :, 5]
    elif metric == 'test/discount_return':
        y = results[:, :, 6]
    else:
        import pdb;pdb.set_trace()

    x = (results[:,:,0].astype(int)+1) * 5
    if env_name in ['Reacher-v2', 'SawyerDoor-v0']:
        x = x[:,::2]
        y = y[:, ::2]
    if smooth:
        for i in range(len(x)):
            x_t, y_t = smooth_reward_curve(x[i], y[i])
            x[i] = x_t
            y[i] = y_t
    xs, ys = pad(x), pad(y)
    assert xs.shape == ys.shape
    if method == 'median':
        plt.plot(xs[0], np.nanmedian(ys, axis=0), label='PPO')
        plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25)
    elif method == 'mean':
        y_mean, y_std = np.mean(ys, axis=0), np.std(ys, axis=0)
        plt.plot(xs[0], y_mean, label='PPO')
        plt.fill_between(xs[0], y_mean - y_std, y_mean + y_std, alpha=0.25)
    else:
        print('no such plot method')
        import pdb;pdb.set_trace()

def plot_line(env_name, xs, paths):
    path = paths[0]
    avg_return = 0
    if 'random' in path:
        avg_return = dataset_return['random'][env_name]
    elif 'medium' in path:
        avg_return = dataset_return['medium'][env_name]
    elif 'expert' in path:
        avg_return = dataset_return['expert'][env_name]
    else:
        print('no such data type')
        import pdb;pdb.set_trace()
    ys = np.zeros_like(xs) + avg_return
    plt.plot(xs, ys, label='Data Return', linestyle=':', color='black')


# Plot datas
def plot_datas(datas, labels, info, key=None, fontsize=16, i=0, j=0, method='mean', more=None):
    # plt.clf()
    title, xlabel, ylabel = info
    max_y = 0
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
        max_y = ys.max() if ys.max() > max_y else max_y

    if 'offline' in labels[0]:
        plot_line(title,xs[0], more)
    else:
        pass
        # plot_ppo(title, key, method=method)
    
    if 'success_rate' in key:
        plt.ylim(0, 1.04 * max_y)
        plt.xlim(0,int(1.02 * xs[0].max()))
    else:
        plt.ylim(0,int(1.05 * max_y))
        plt.xlim(0,int(1.02 * xs[0].max()))
    
    plt.title(title, fontsize=fontsize-2)
    if i == row - 1:
        plt.xlabel(xlabel, fontsize=fontsize-2)
    if j == 0:
        plt.ylabel(ylabel, fontsize=fontsize-2)
    # if j == 0:
    # plt.legend(fontsize=fontsize-6) #, loc=4, bbox_to_anchor=(0.5, 0.06, 0.5, 0.5))
    plt.xticks(fontsize=fontsize-4)
    plt.yticks(fontsize=fontsize-4)

def plot_main(dirs, labels, info, key='test/success_rate', filename='progress.csv', save_dir='./test.png', x_time='epoch'):
    plt.figure(dpi=300, figsize=(5,4))
    sns.set(style='whitegrid')
    datas = load_datas(dirs, key, filename, x_time)

    plot_datas(datas, labels, info, key=key)
    plt.subplots_adjust(left=0.15, right=0.98, bottom=0.14, top=0.92, hspace=0.3, wspace=0.15)
    plt.savefig(save_dir)

def subplot_main(pic_dirs, labels, infos, row=1, key='test/success_rate',\
                    filename='progress.csv', save_dir='./test.png', x_time='epoch'):
    n = len(pic_dirs)  # number of pics
    col = math.ceil(n / row)
    
    if 'offline' in labels[0] or 'DRW' in labels[1] or 'clip' in labels[0]:
        fig, axes = plt.subplots(row, col, figsize=(3 * col, 3.6))
        fig.subplots_adjust(left=0.035, right=0.992, bottom=0.15, top=0.805, hspace=0.25, wspace=0.15)
    else:
        fig, axes = plt.subplots(row, col, figsize=(3.6 * col, 3 * row))  #3.6
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.1, top=0.87, hspace=0.3, wspace=0.22)  
    sns.set_style('whitegrid', {'grid.linestyle': '--'})

    titles, x_label, y_label = infos
    xy_label = [x_label, y_label]
    for i in range(row):
        for j in range(col):
            idx = i * col + j 
            if idx + 1 > n:
                break
            
            plt.subplot(row, col, idx+1)
            datas = load_datas(pic_dirs[idx], key, filename, x_time)
            plot_datas(datas, labels, [titles[idx], *xy_label], key=key, i=i, j=j, more=pic_dirs[idx])
    if 'offline' in labels[0]:
        legend = labels + ['Averge Return of Dataset']
    elif 'DRW' in labels[1] or 'clip' in labels[0]:
        legend = labels
    else:  
        legend = labels + ['PPO'] 
    fig.legend(legend, loc='upper center', ncol=6, fontsize=13)
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
   


    ################################## Iterated results
    # # subplot figures
    # save_dir = './iterated.pdf'
    # prefix_dir = '/Users/yangrui/Desktop/logs_weighted/latest/'
    
    # # temp = 'gamma_exp_adv_gcsl/'
    # # path_list = ['', 'no_target_new_param/', 'ordered_buffer_5e4/','small_buffer1e4/', 'small_buffer2e3/']
    # # path_list = [x + temp for x in path_list]
    # #  'gamma_gcsl/', 'exp_adv_clip10_gcsl/',
    # path_list = [ 'gamma_exp_adv_clip10_gcsl/', 'gcsl/',  'her_td3_2penalty_bigbatch2',  'td3_2/'] 
    # pic_title = ['Point2DLargeEnv-v1', 'Point2D-FourRoom-v1', 'SawyerReachXYZEnv-v1', 'FetchReach-v1', 'Reacher-v2', 'SawyerDoor-v0']  
  
    # pic_dirs = [[[] for _ in range(len(path_list))] for _ in range(len(pic_title))]
    # for i in range(len(pic_title)):
    #     for j in range(len(path_list)):
    #         pic_dirs[i][j] = prefix_dir + path_list[j] + '/' + pic_title[i]
 
    # infos = [pic_title, 'Episodes',  'Average Return'] #'Mean Success Rate'
    # row = 2
    # # legend = ['origin', 'new param/no target', 'ordered buffer 5e4','ordered buffer 1e4', 'ordered buffer 2e3']
    # #  'GCSL+gamma', 'GCSL+exp_adv clip10'
    # legend = ['WGCSL(Ours)', 'GCSL', 'TD3+HER', 'TD3' ]   
    # subplot_main(pic_dirs, legend, infos, row, key='test/return', save_dir=save_dir, x_time='episode')

# ##### offline
 # subplot figures
    # dataset_return = {
    #     'random':{
    #         'Point2DLargeEnv-v1': 1.33,
    #         'Point2D-FourRoom-v1':1.32,
    #         'SawyerReachXYZEnv-v1': 1.25,
    #         'FetchReach-v1':0.71,
    #         'Reacher-v2':2.26,
    #         'SawyerDoor-v0':4.30
    #     },
    #     'medium':{
    #         'Point2DLargeEnv-v1': 30.99,
    #         'Point2D-FourRoom-v1':27.30,
    #         'SawyerReachXYZEnv-v1': 28.39,
    #         'FetchReach-v1':35.71,
    #         'Reacher-v2':20.45,
    #         'SawyerDoor-v0':24.02
    #     },
    #     'expert':{
    #         'Point2DLargeEnv-v1': 32.22,
    #         'Point2D-FourRoom-v1':29.11,
    #         'SawyerReachXYZEnv-v1': 30.93,
    #         'FetchReach-v1':36.69,
    #         'Reacher-v2':27.56,
    #         'SawyerDoor-v0':27.01
    #     },
    # }
    # save_dir = './offline_more.png'
    # prefix_dir = '/Users/yangrui/Desktop/logs_offline/training/'
    
    # type_data = 'expert'
    # save_dir = save_dir.replace('.png', '_' + type_data + '.pdf')
    # path_list = [ 'gamma_exp_adv_clip10_', 'gcsl_','marvil/exp_adv_clip10_','bc_','bcq_more_' ] 
    # path_list = [p + type_data + '/' for p in path_list]
    # pic_title = ['Point2DLargeEnv-v1', 'Point2D-FourRoom-v1',  'SawyerReachXYZEnv-v1','FetchReach-v1', 'Reacher-v2', 'SawyerDoor-v0']  
    
  
    # pic_dirs = [[[] for _ in range(len(path_list))] for _ in range(len(pic_title))]
    # for i in range(len(pic_title)):
    #     for j in range(len(path_list)):
    #         pic_dirs[i][j] = prefix_dir + path_list[j] + '/' + pic_title[i]
 
    # infos = [pic_title, 'Training Steps', 'Average Return'] #'Median Success Rate']
    # row = 1
    # legend = [ 'offline WGCSL', 'offline GCSL', 'Goal MARVIL', 'Goal BC', 'Goal BCQ']   
    # subplot_main(pic_dirs, legend, infos, row, key='test/return', save_dir=save_dir, x_time='episode')



# ######################## ablation study
# save_dir = './ablations_rate.pdf'
# prefix_dir = '/Users/yangrui/Desktop/logs_weighted/latest/'

# # temp = 'gamma_exp_adv_gcsl/'
# # path_list = ['', 'no_target_new_param/', 'ordered_buffer_5e4/','small_buffer1e4/', 'small_buffer2e3/']
# # path_list = [x + temp for x in path_list]
# #  , ,
# path_list = ['gcsl/', 'gamma_gcsl/', 'exp_adv_clip10_gcsl/',  'gamma_exp_adv_clip10_gcsl/'] 
# pic_title = ['Point2DLargeEnv-v1', 'Point2D-FourRoom-v1', 'SawyerReachXYZEnv-v1', 'FetchReach-v1', 'Reacher-v2', 'SawyerDoor-v0']  

# pic_dirs = [[[] for _ in range(len(path_list))] for _ in range(len(pic_title))]
# for i in range(len(pic_title)):
#     for j in range(len(path_list)):
#         pic_dirs[i][j] = prefix_dir + path_list[j] + '/' + pic_title[i]

# infos = [pic_title, 'Episodes', 'Mean Success Rate' ] #'Average Return'
# row = 1
# # legend = ['origin', 'new param/no target', 'ordered buffer 5e4','ordered buffer 1e4', 'ordered buffer 2e3']
# #  'GCSL+gamma', 'GCSL+exp_adv clip10'
# legend = ['GCSL', 'GCSL + DRW', 'GCSL + GEAW','WGCSL']   
# subplot_main(pic_dirs, legend, infos, row, key='test/success_rate', save_dir=save_dir, x_time='episode')


# ######################## ablation study
save_dir = './clip_exp_adv.pdf'
prefix_dir = '/Users/yangrui/Desktop/logs_weighted/latest/'

#  , ,
path_list = ['gamma_exp_adv_gcsl/', 'gamma_exp_adv_clip5_gcsl/', 'gamma_exp_adv_clip10_gcsl/'] 
pic_title = ['Point2DLargeEnv-v1', 'Point2D-FourRoom-v1', 'SawyerReachXYZEnv-v1', 'FetchReach-v1', 'Reacher-v2', 'SawyerDoor-v0']  

pic_dirs = [[[] for _ in range(len(path_list))] for _ in range(len(pic_title))]
for i in range(len(pic_title)):
    for j in range(len(path_list)):
        pic_dirs[i][j] = prefix_dir + path_list[j] + '/' + pic_title[i]

infos = [pic_title, 'Episodes',  'Average Return'] #'Mean Success Rate'
row = 1
# legend = ['origin', 'new param/no target', 'ordered buffer 5e4','ordered buffer 1e4', 'ordered buffer 2e3']
#  'GCSL+gamma', 'GCSL+exp_adv clip10'
legend = ['WGCSL w/o clip', 'WGCSL clip 5', 'WGCSL clip 10']   
subplot_main(pic_dirs, legend, infos, row, key='test/return', save_dir=save_dir, x_time='episode')
