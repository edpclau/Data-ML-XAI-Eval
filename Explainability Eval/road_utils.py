
#Evals are too big in RAM so we must chunk the analysis and import of them.
#We will define functions that will plot parts of the evals and that will output summarized results.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, rankdata


#Function to get the correlation between the ground truth and the shap values
def get_kendall(values, groundnp):
        # vals = get_all_values(values[3]) #road results is a list of lists where the list at index 3 are the shap values
        vals = values[3]
        #vals dimensions (repeats, features, samples)
        vals = rankdata(vals, axis = 1)

        try:
                corr = np.array([kendalltau(groundnp, vals[:,:,i])[0] for i in range(vals.shape[2])])
        except:
                print('Error in kendalltau')
                print(vals)
                print(vals.shape)
                return

        if np.all(np.isnan(corr)):
            return np.array([0, 0])

        return [corr.mean(where = ~np.isnan(corr)), corr.std(where = ~np.isnan(corr))] #returns average kendall's tau correlation and std


def get_distance(values, groundnp):
        vals = values[3]
        #vals dimensions (repeats, features, samples)
        vals = rankdata(vals, axis = 1)
        dist = np.array([np.linalg.norm(groundnp - vals[:,:,i]) for i in range(vals.shape[2])])
        return [dist.mean(), dist.std()] #returns average euclidean distance and std

#Function to get the similarity score across all repeats
def similarity_score(results):
        return results[3].std(2).mean()  #output the mean variance of all features.


#ROAD score function
def road_score(results):
        results = np.dstack([results[0:3]])
        auc = np.trapz(results, axis = 2)
        #0 = Top, 1 = Bottom, 2  = Random
        results = np.where(auc[2] >= auc[0],  (auc[2] - auc[0])/auc[2], (auc[2] - auc[0])/auc[0])
        mean = results.mean(axis=1)
        sd = results.std(axis=1)
        return np.array([mean, sd])


def road_score_alt1(results):
        #0 = Top, 1 = Bottom, 2  = Random
        score_bottom = get_score_bottom(results)
        score_top = get_score_top(results)

        results = np.mean([score_bottom, score_top], axis = 0)


        return results

#Function to rank the output of road and rank each step
def get_road_top(results):
       #0 = Top, 1 = Bottom, 2  = Random
       return results[0].mean(axis = 2).T

def get_road_bottom(results):
        #0 = Top, 1 = Bottom, 2  = Random
        return results[1].mean(axis = 2).T

def get_road_random(results):
        #0 = Top, 1 = Bottom, 2  = Random
        return results[2].mean(axis = 2).T

#Function to get the score of the top and bottom k features
def get_road_spearman(results):
        #0 = Top, 1 = Bottom, 2  = Random
        top = results[0].mean(axis = 2).T
        bottom = results[1].mean(axis = 2).T
        #Get spearman's correlation of top and bottom
        return np.array([kendalltau(top[:,i], bottom[:,i])[0] for i in range(top.shape[1])])


def get_score_top(results):
        #0 = Top, 1 = Bottom, 2  = Random
        top = np.array(results[0]).mean(axis=2).T
        random = np.array(results[2]).mean(axis=2).T
        #Get auc of top and random
        top_auc = np.trapz(top, axis = 0, x = np.linspace(0, 1, num = top.shape[0]))
        random_auc = np.trapz(random, axis = 0, x = np.linspace(0, 1, num = random.shape[0]))
        #Get the score
        results = np.where(top_auc >= random_auc,
                           (random_auc/top_auc) - 1,
                           1 - (top_auc/random_auc)
                          )
        return results

def get_score_bottom(results):
        #0 = Top, 1 = Bottom, 2  = Random
        bottom = np.array(results[1]).mean(axis=2).T
        random = np.array(results[2]).mean(axis=2).T
        #Get auc of bottom and random
        bottom_auc = np.trapz(bottom, axis = 0, x = np.linspace(0, 1, num = bottom.shape[0]))
        random_auc = np.trapz(random, axis = 0, x = np.linspace(0, 1, num = random.shape[0]))
        results = np.where(bottom_auc >= random_auc, 
                           1 - (random_auc/bottom_auc), 
                           (bottom_auc/random_auc) - 1
        )
        return results


def get_top(results):
        results = np.dstack([results[0:3]])
        #0 = Top, 1 = Bottom, 2  = Random
        mean = np.trapz(np.array(results[0]).mean(axis=2).T, axis=0, x = np.linspace(0, 1, num = results[0].shape[0]))
        return mean

def get_bottom(results):
        results = np.dstack([results[0:3]])
        #0 = Top, 1 = Bottom, 2  = Random
        mean = np.trapz(np.array(results[1]).mean(axis=2).T, axis=0, x = np.linspace(0, 1, num = results[1].shape[0]))
        return mean


def get_random(results):
        results = np.dstack([results[0:3]])
        #0 = Top, 1 = Bottom, 2  = Random
        mean = np.trapz(np.array(results[2]).mean(axis=2).T, axis=0, x = np.linspace(0, 1, num = np.array(results[2]).mean(axis=2).T.shape[0]))
        return mean

## Get average road results
#Arguments
#results: the output of road
def get_avg_results(results):
        results = np.dstack([results[0:3]])
        mean = results.mean(axis=3)
        sd = results.std(axis=3)
        return np.array([mean, sd])


## plot the outputs of road ##
#Arguments
#results: the output of road
#model: the model used
#explainer: the explainer used
#dataset: the dataset used
def road_plot(results, model, explainer, dataset):
    method = 1
    top_mean = results[0].mean(axis = 2).T[:, method]
    bottom_mean = results[1].mean(axis = 2).T[:, method]
    random_mean = results[2].mean(axis = 2).T[:, method]

    road_score_auc = np.round(road_score_alt1(results)[method]*100, decimals= 2)
    road_bottom = np.round(get_score_bottom(results)[method]*100, decimals= 2)
    road_top = np.round(get_score_top(results)[method]*100, decimals = 2)


    road_score_spear = np.round(get_road_spearman(results)[method], decimals=3)


    x_axis = np.linspace(0, 100, num = top_mean.shape[0])

    #Plotting
    # Make 2 subplots
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=600)

    # Subplot 1: Plot the mean and standard deviation of the top, bottom, and random and fill in the area between them
    ax.plot(x_axis, top_mean, label = 'Mask Most Important First (MMIF)')
    ax.plot(x_axis, bottom_mean, label = 'Mask Least Important First (MLIF)')
    ax.plot(x_axis, random_mean, label = 'Mask Randomly')
    ax.hlines(y = 0.5, xmin = 0, xmax = 100, color = 'black', linestyle = '--')
    ax.scatter(x_axis, top_mean)
    ax.scatter(x_axis, bottom_mean)
    ax.scatter(x_axis, random_mean)
    ax.fill_between(x_axis, y1 = random_mean, y2 = bottom_mean, color = 'purple', alpha = 0.1, label = f'Area Difference = {road_bottom}%')
    ax.fill_between(x_axis, y1 = top_mean, y2 = random_mean, color = 'brown', alpha = 0.1, label = f'Area Difference = {road_top}%')
    ax.plot([], [], ' ', label = f'Average Area Difference = {road_score_auc}%')
    # ax.plot([], [], ' ', label = f'Rank Correlation MMIF vs MLIF = {road_score_spear}')
    ax.set_ylabel('AUROC')
    ax.set_title(f'Model: {model} | Explainer: {explainer} | Dataset: {dataset}')

    # Make legend prettier
    ax.legend( loc='upper right')
    plt.xlabel('Percentage of Features Masked')