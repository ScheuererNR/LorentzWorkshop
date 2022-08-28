
import numpy as np
import pandas as pd



plt.ion()

ncat = 20


#----------------------------------------------------------------------------------------
#  1. Create some data set to test the verification routines
#   Since I currently can't work on SRC, I'm using a data set from a different project.
#   The assumption is that we have a pd.DataFrame with one-hot encoded observations
#   (below normal, normal, above normal) and another one with predicted probabilities


ilead = 0
igpt = 100
pp_method = 'ANN'

f1 = np.load(f'/home/michael/Desktop/CalifAPCP/results/scores-ann_week{ilead+2}.npz')
exc33p = f1['exc33p'][:,:,igpt]
exc67p = f1['exc67p'][:,:,igpt]
pot33p = f1[f'pot33p{pp_method}'][:,:,igpt]
pot67p = f1[f'pot67p{pp_method}'][:,:,igpt]
f1.close()

f2 = np.load('/home/michael/Desktop/CalifAPCP/data/mod_precip_cal.npz')
mod_dates = f2['dates_ord'][:,:,7*ilead+6]
f2.close()

ndts, nyrs = mod_dates.shape
datelist = [pd.Timestamp.fromordinal(int(mod_dates.flatten()[i])) for i in range(ndts*nyrs)]

obs_binary = pd.DataFrame(index=datelist, columns=['below normal','normal','above normal'])
obs_binary['below normal'] = 1.-exc33p.flatten()
obs_binary['normal'] = (exc33p-exc67p).flatten()
obs_binary['above normal'] = exc67p.flatten()

fcst_prob = pd.DataFrame(index=datelist, columns=['below normal','normal','above normal'])
fcst_prob['below normal'] = 1.-pot33p.flatten()
fcst_prob['normal'] = (pot33p-pot67p).flatten()
fcst_prob['above normal'] = pot67p.flatten()





#----------------------------------------------------------------------------------------
#  2. Define function to compute Brier skill score and apply to data set


def brier_score(y, x):
    return ((x-y)**2).sum(1).mean(0)

def brier_skill_score(y, x, x0):
    return 1. - brier_score(y,x) / brier_score(y,x0)

def l1yocv_climatology(y):
    prob = pd.DataFrame(0.0, index=y.index, columns=y.columns)
    years = y.index.year.unique()
    for yr in years:
        prob_yr = y.loc[y.index.year!=yr].mean(0)
        prob.loc[y.index.year==yr] = prob.loc[y.index.year==yr].add(prob_yr)
    return prob


clm_prob = l1yocv_climatology(obs_binary)

brier_skill_score(obs_binary, fcst_prob, clm_prob)






#----------------------------------------------------------------------------------------
#  3. Define function to plot reliability diagrams for the three categories


def reliability_diagram(y, x, nbins=11, nmin=50):
    fig = plt.figure(figsize=(14,4.5))
    for icat in range(3):
        category = x.columns[icat]
        xy_cat = pd.concat([y[category].rename('obs'),x[category].rename('prob'),(x[category]*(nbins-1)).round().rename('stratum')], axis=1)
        freq = xy_cat.groupby(['stratum']).size()
        relia = xy_cat.groupby(['stratum']).mean()
        relia[freq<nmin] = np.nan
        ax = fig.add_subplot(1,3,1+icat)
        rel = plt.plot(relia['prob'].dropna(), relia['obs'].dropna(), '-o', c='royalblue')
        plt.plot([0,1], [0,1], c='k')
        plt.axvline(0.33, c='k', ls=':', lw=1, ymin=0.05, ymax=0.58)
        plt.title(f'Reliability for "{category}"\n',fontsize=14)
        ins = ax.inset_axes([0.03,0.70,0.35,0.25])
        ins.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        ins.set_xlabel('Frequency of usage', fontsize=11)
        ins.bar(np.arange(nbins), freq.reindex(np.arange(nbins)).fillna(0), 0.6, color='royalblue')
    plt.tight_layout()


reliability_diagram(obs_binary, fcst_prob)
plt.show()





