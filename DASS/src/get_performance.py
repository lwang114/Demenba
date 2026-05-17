import pickle
import matplotlib.pyplot as plt
import numpy as np

def prettifyAxes(ax):
    ax.patch.set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    plt.tick_params(axis='y',direction='out', right=False)
    plt.tick_params(axis='x',direction='out', top=False)

models = {#1: ('tiny', False, 'exp'), 
          1: ('medium', False, 'exp_WCE_macro_f1'), 
          2: ('medium', True, 'exp_CE_macro_f1'),
          3: ('medium', True, 'exp_WCE_macro_f1'), 
          4: ('medium', True, 'exp_WCE_macro_f1_audioset'), 
          5: ('medium', True, 'exp_BCE_macro_f1'), 
          #6: ('small', True, 'exp_BCE_no_audioset'),
          #7: ('medium', True, 'exp')
          }
performance = {}
performance_small = {}
bwidth = 0.8
colors = {1: [el/255. for el in [51,204,255]], 
          2: [el/255. for el in [153,102,255]], 
          3: [el/255. for el in [119,225,119]],
          4: 'orange',
          5: 'pink',
          6: 'turquoise',
          7: 'magenta'}
model_names = [#'Tiny (no imnet)', 
                '(WCE, no imnet)', 
               '(CE)',
'(WCE)', '(WCE, audioset)', '(BCE)',] #'Small (BCE no audioset)', 'Medium']
for model, (size, imnet, expdir) in models.items():
    accs = []
    accs_small = []
    for i in range(1, 6):
        with open(f'../egs/audioset/{expdir}/DASS-{size}-{i}-balanced-p{imnet}-b12-lr0.0001-kdFalse-kddkldiv-dt1.0/progress.pkl', 'rb') as f:
            progress = pickle.load(f)
        with open(f'../egs/audioset/{expdir}/DASS-{'small'}-{i}-balanced-p{imnet}-b12-lr0.0001-kdFalse-kddkldiv-dt1.0/progress.pkl', 'rb') as f:
            progress_small = pickle.load(f)
        accs.append(progress[-1][-3])
        accs_small.append(progress_small[-1][-3])
    print(accs)
    performance[model] = accs
    performance_small[model] = accs_small

plt.figure(figsize=(12, 4))
fig, ax = plt.subplots(layout='constrained')
for i, model in zip(np.arange(len(models))-bwidth/2.,models):
    ax.bar(i-bwidth/4., np.mean(performance_small[model]), bwidth/2.,
                color=colors[model], lw=0, 
                yerr=np.nanstd(performance_small[model])/np.sqrt((~np.isnan(performance_small[model])).sum()),
                error_kw={'ecolor': 'k', 'capsize': 0, 'lw': 3})
    ax.bar(i+bwidth/4., np.mean(performance[model]), bwidth/2.,
                color=colors[model], lw=0, 
                yerr=np.nanstd(performance[model])/np.sqrt((~np.isnan(performance[model])).sum()),
                error_kw={'ecolor': 'k', 'capsize': 0, 'lw': 3})
plt.ylim(0,1); plt.xlim(-1,len(models) - 1 + .5)
prettifyAxes(plt.gca())
plt.ylabel('F1 score', {'fontsize': 12})
plt.xticks([-0.4] + [i + 0.6 for i in range(len(models)-1)], model_names, rotation=0)
#plt.legend(['H', 'Kell 2018', 'Nat. model', 'CV model'])
plt.title('Average class-1 F1 score by model', {'fontsize': 14})

plt.savefig('5_model_macro_f1_medium.png') 

means = [f'| {np.mean(performance[model]):.4f} ' for model in models]
variances = [f'| {np.std(performance[model])**2:.4f} ' for model in models]
mean_str = ''
for mean in means:
    mean_str += mean
var_str = ''
for var in variances:
    var_str += var
print('\n')
print(' '  + '        |' + '  blue  |' + ' purple |' + '  lime  |' + ' orange |' + '  pink  ')
print('-'*55)
print('   mean ', mean_str)
print('-'*55)
print('    var ', var_str)
print('\n')



    