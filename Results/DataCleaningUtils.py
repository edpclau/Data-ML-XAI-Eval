
#Evals are too big in RAM so we must chunk the analysis and import of them.
#We will define functions that will plot parts of the evals and that will output summarized results.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, rankdata, pearsonr, spearmanr
import joblib
import os
import pandas as pd


#Function to get the correlation between the ground truth and the shap values
def get_kendall(values, groundnp):
        # vals = get_all_values(values[3]) #road results is a list of lists where the list at index 3 are the shap values
        vals = values[3]
        # vals[:, 3, :] = np.nan_to_num(vals[:, 3, :], nan = 0)
        #vals dimensions (repeats, features, samples)
        # vals = rankdata(vals, axis = 1)

        try:
                corr = np.array([kendalltau(groundnp, vals[:,:,i][0])[0] for i in range(vals.shape[2])])
        except:
                print('Error in kendalltau')
                print('Groundnp')
                print(groundnp)
                print(groundnp.shape)
                print('Vals')
                print(vals)
                print(vals.shape)
                print('\n')
                return

        if np.any(np.isnan(corr)):
                print('Error in kendalltau')
                print('Groundnp')
                print(groundnp)
                print(groundnp.shape)
                print('Vals')
                print(vals)
                print(vals.shape)
                return np.array([0, 0])

        return [corr, corr] #returns average kendall's tau correlation and std

#Function to check if the explainer passes the monotonicity test. Which means that features that aren't important should have a shap_value of 0
def get_noise(values):
        # vals = get_all_values(values[3]) #road results is a list of lists where the list at index 3 are the shap values
        vals = values[3]
        return np.nan_to_num(np.nanmean(vals[:, 3, :]), nan = 0)

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
        mean = results
        sd = results
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
       return results[0]

def get_road_bottom(results):
        #0 = Top, 1 = Bottom, 2  = Random
        return results[1]

def get_road_random(results):
        #0 = Top, 1 = Bottom, 2  = Random
        return results[2]

#Function to get the score of the top and bottom k features
def get_road_spearman(results):
        #0 = Top, 1 = Bottom, 2  = Random
        top = results[0].mean(axis = 2).T
        bottom = results[1].mean(axis = 2).T
        #Get spearman's correlation of top and bottom
        return np.array([kendalltau(top[:,i], bottom[:,i])[0] for i in range(top.shape[1])])


def get_score_top(results):
        #0 = Top, 1 = Bottom, 2  = Random
        top = np.array(results[0])
        random = np.array(results[2])
        #Get auc of top and random
        top_auc = np.trapz(top, axis = 1, x = np.linspace(0, 1, num = top.shape[1]))
        random_auc = np.trapz(random, axis = 1, x = np.linspace(0, 1, num = random.shape[1]))
        #Get the score
        results = np.where(top_auc >= random_auc,
                           (random_auc/top_auc) - 1,
                           1 - (top_auc/random_auc)
                          )
        return results

def get_score_bottom(results):
        #0 = Top, 1 = Bottom, 2  = Random
        bottom = np.array(results[1])
        random = np.array(results[2])
        #Get auc of bottom and random
        bottom_auc = np.trapz(bottom, axis = 1, x = np.linspace(0, 1, num = bottom.shape[1]))
        random_auc = np.trapz(random, axis = 1, x = np.linspace(0, 1, num = random.shape[1]))
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
        mean = np.trapz(np.array(results[2]), axis=1, x = np.linspace(0, 1, num = np.array(results[2]).shape[1]))
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


## Function to clean up the results of road ##
#Arguments
#models: list of models used
#explainers: list of explainers used
#datasets: list of datasets used
#path: path to the results
def clean_road_results(models, explainers, datasets, repeats, ground, dir_path):
        #Get the score and similarity for every file, model, and explainer
        score = {}
        score_bottom = {}
        score_top = {}
        similarity = {}
        kendall = {}
        random = {}
        noise = {}


        for file in datasets:
                score_ = {}
                score_bottom_ = {}
                score_top_ = {}
                similarity_ = {}
                kendall_ = {}
                random_ = {}
                noise_ = {}
                
                #Get the ground truth for a particular file
                groundnp = ground.loc[ground.Dataset == file].Rank.values
        
                for model in models:

                        _score_ = {}
                        _score_bottom_ = {}
                        _score_top_ = {}
                        _similarity_ = {}
                        _kendall_ = {}
                        _random_ = {}
                        _noise_ = {}
                        
                        #Create a dictionary to store the results
                        for xai in explainers:

                                for rep in repeats:
                                        path = f'{dir_path}/{file}_{model}_{xai}_{rep}.joblib'
                                        
                                        if not os.path.isfile(path):
                                                does_not_exist = True
                                                # print(f'{path} does not exist')
                                                continue
                                        
                                        
                                        top, bottom, rand, shap_values = joblib.load(path)

                                        if shap_values is None:
                                                does_not_exist = True
                                                continue

                                        does_not_exist = False

                                        #stack the results
                                        if rep == 0:
                                                top_ = top
                                                bottom_ = bottom
                                                rand_ = rand
                                                shap_values_ = shap_values
                                        else:
                                                top_ = np.dstack([top_, top])
                                                bottom_ = np.dstack([bottom_, bottom])
                                                rand_ = np.dstack([rand_, rand])
                                                shap_values_ = np.dstack([shap_values_, shap_values])
                        
                        
                                if does_not_exist:
                                        continue

                                evals = [top_, bottom_, rand_, shap_values_]
                                
                                #Get the score for a particular file, model, and explainer
                                _score_[xai] = road_score_alt1(evals)
                                _score_bottom_[xai] = get_score_bottom(evals)
                                _score_top_[xai] = get_score_top(evals)
                                _similarity_[xai] = similarity_score(evals)
                                _kendall_[xai] = get_kendall(evals, groundnp)
                                _random_[xai] = get_random(evals)
                                _noise_[xai] = get_noise(evals)
                                
                                #  #Plot the results and Save the figure
                                #     road_plot(evals, model, xai, file)
                                #     plt.savefig(f'/Users/eddie/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Research/XAI method performacne when Explainaing the PORT Dataset/Figures/Simulated/Full Figure/ROAD/{file}_{model}_{xai}.pdf', dpi=600, bbox_inches = 'tight', transparent = False)
                                #     plt.close()

                        if does_not_exist:
                                continue
                        
                        score_[model] = _score_
                        score_bottom_[model] = _score_bottom_
                        score_top_[model] = _score_top_
                        similarity_[model] = _similarity_
                        kendall_[model] = _kendall_
                        random_[model] = _random_
                        noise_[model] = _noise_
                
                if does_not_exist:
                        continue

                score[file] = score_
                score_bottom[file] = score_bottom_
                score_top[file] = score_top_
                similarity[file] = similarity_
                kendall[file] = kendall_
                random[file] = random_
                noise[file] = noise_

        return score, score_bottom, score_top, similarity, kendall, random, noise

# Turn the results into a dataframe
def results_to_dataframes(score, score_bottom, score_top, similarity, kendall, random, noise):
                ## Make a Data Frame of ROAR Score
        df_score_mean = {}
        for file, models in score.items():
                df_models = {}
                for model, exp in models.items():
                        df_exp = {}
                        for explainer, roar_results in exp.items():   
                                df_exp[explainer] = pd.Series(roar_results[3])
                        df_models[model] = pd.concat(df_exp)
                df_score_mean[file] = pd.DataFrame(df_models)
        df_score_mean = pd.concat(df_score_mean)

        ## Make a Data Frame for ROAR Score Bottom
        df_score_bottom_mean = {}
        for file, models in score_bottom.items():
                df_models = {}
                for model, exp in models.items():
                        df_exp = {}
                        for explainer, roar_results in exp.items():   
                                df_exp[explainer] = pd.Series(roar_results[3])
                        df_models[model] = pd.concat(df_exp)
                df_score_bottom_mean[file] = pd.DataFrame(df_models)
        df_score_bottom_mean = pd.concat(df_score_bottom_mean)

        ## Make a Data Frame for ROAR Score Top
        df_score_top_mean = {}
        for file, models in score_top.items():
                df_models = {}
                for model, exp in models.items():
                        df_exp = {}
                        for explainer, roar_results in exp.items():   
                                df_exp[explainer] = pd.Series(roar_results[3])
                        df_models[model] = pd.concat(df_exp)
                df_score_top_mean[file] = pd.DataFrame(df_models)
        df_score_top_mean = pd.concat(df_score_top_mean)



        #Data frame of the score
        df_sim_std = {}
        for file, models in similarity.items():
                df_models = {}
                for model, exp in models.items():
                        df_exp = {}
                        for explainer, roar_results in exp.items():
                                df_exp[explainer] = roar_results
                        df_models[model] = df_exp
                df_sim_std[file] = pd.DataFrame(df_models).T
        df_sim_std = pd.concat(df_sim_std)


        #Data frame of kendall's tau
        df = {}
        for file, models in kendall.items():
                df_models = {}
                for model, exp in models.items():
                        df_exp = {}
                        for explainer, roar_results in exp.items():
                                df_exp[explainer] = pd.Series(roar_results[0])
                        df_models[model] = pd.concat(df_exp)
                df[file] = pd.DataFrame(df_models)
        df_kendall = pd.concat(df)


        # #Data frame of distance
        # df = {}
        # for file, models in distance.items():
        #     df_models = {}
        #     for model, exp in models.items():
        #         df_exp = {}
        #         for explainer, roar_results in exp.items():
        #             df_exp[explainer] = roar_results[0]
        #         df_models[model] = df_exp
        #     df[file] = pd.DataFrame(df_models).T
        # df_distance = pd.concat(df)


        #Data frame of random
        df = {}
        for file, models in random.items():
                df_models = {}
                for model, exp in models.items():
                        df_exp = {}
                        for explainer, roar_results in exp.items():
                                df_exp[explainer] = pd.Series(roar_results[3])
                        df_models[model] = pd.concat(df_exp)
                df[file] = pd.DataFrame(df_models)
        df_random = pd.concat(df)

        #Data frame of noise
        df = {}
        for file, models in noise.items():
                df_models = {}
                for model, exp in models.items():
                        df_exp = {}
                        for explainer, roar_results in exp.items():
                                df_exp[explainer] = pd.Series(np.repeat(roar_results, 10))
                        df_models[model] = pd.concat(df_exp)
                df[file] = pd.DataFrame(df_models)
        df_noise = pd.concat(df)

        return df_score_mean, df_score_bottom_mean, df_score_top_mean, df_sim_std, df_kendall, df_random, df_noise