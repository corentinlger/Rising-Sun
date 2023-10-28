import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python.summary.summary_iterator import summary_iterator

def get_logs_values(logdir, models, nb_seeds):
    """
    Download data from Tensorboard log files and returns the mean episodic reward array, the std episodic reward array
    and the timesteps array associated in order to plot and observe these results for the desired models.

    :param logdir: (str) log directory
    :param models: ([str]) models names list
    :param nb_seeds: (int) nb of random seeds experiments
    :return: (np.ndarray, np.ndarray, np.ndarray) models_mean_arrays, models_std_arrays, timesteps
    """

    # Downloading the data from the logs directory in a pandas DataFrame
    dfs = []

    for dir in os.listdir(logdir):
        if os.path.isdir(os.path.join(logdir, dir)):
            for model_name in models :
                if dir.startswith(model_name):
                    model = model_name
                    for i in range(5):
                        if dir.endswith(f"{i}_1"):
                            seed = i

                            subdir = os.path.join(logdir, dir)
                            files = [os.path.join(subdir, file) for file in os.listdir(subdir)]
                            data = []

                            for filepath in files:
                                for event in summary_iterator(filepath):
                                    for value in event.summary.value:
                                        if value.tag =='rollout/ep_rew_mean':
                                            data.append([event.step, value.simple_value, model, seed])

                            df = pd.DataFrame(data, columns=['step', 'ep_rew_mean', 'model', 'seed'])
                            dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Transforming the pandas data into numpy arrays of mean and std reward for each model

    models_tot_arrays = {}
    models_mean_arrays = {}
    models_std_arrays = {}

    for model in models:
        arr = []
        for seed in range(nb_seeds):
            serie = df_all.loc[(df_all['model'] == model) & (df_all['seed'] == seed), 'ep_rew_mean']
            serie = serie.reset_index(drop=True)  # reset index
            if len(serie) == 0 :
                print(f"data missing for {model} seed {seed}")
            else:
                arr.append(serie)


        df = pd.concat(arr, axis=1)
        nparr = df.to_numpy()

        models_tot_arrays[model] = nparr

        # calculate the mean and the std of this model rewards:
        models_mean_arrays[model] = np.mean(nparr, axis=1)

        models_std_arrays[model] = np.std(nparr, axis=1)


    timesteps = df_all.loc[(df_all['model' ]==model) & (df_all['seed' ]==seed), 'step']
    timesteps = np.array(timesteps, dtype=np.float64)

    return models_mean_arrays, models_std_arrays, timesteps


def plot_training_results(logdir, models):
    """
    Matplotlib plot of the models performance during training
    :param logdir : (str) log directory
    :param models : [(str)] models tested
    """
    models_mean_arrays, models_std_arrays, timesteps = get_logs_values(logdir, models)

    plt.figure(figsize=(15, 6))
    plt.title(f"Evolution of mean episode reward on {logdir}_{models}")
    plt.xlabel('Steps')
    plt.ylabel('Mean episode reward')

    for model in models:
        std = models_std_arrays[model]
        plt.plot(timesteps, models_mean_arrays[model], label=model)
        plt.fill_between(timesteps, models_mean_arrays[model] - models_std_arrays[model],
                         models_mean_arrays[model] + models_std_arrays[model], alpha=0.2)

    plt.legend()
    plt.show()

def plot_evaluation_results(rl_player_wins_list, bot_player_wins_list):
    """
    Matplotlib plot of the model performance against a bot
    :param rl_player_wins_list : [(int)] Evolution of the rl player wins
    :param bot_player_wins_list : [(int)] Evolution of the rl player wins
    """
    x = np.linspace(0, len(rl_player_wins_list), len(rl_player_wins_list))

    plt.figure(figsize=(15, 5))
    plt.title("Number of wins between trained RL and Heuristic Player")
    plt.xlabel("Game number")
    plt.ylabel("nb wins")

    plt.plot(x, rl_player_wins_list, label='rl_player')
    plt.plot(x, bot_player_wins_list, label='bot_player')

    plt.legend()
    plt.show()

def create_saving_directories(exp_name, fights_per_game, bot_behavior, verbose=False):
    experiment_name = f"{exp_name}_" if exp_name else ""
    experiment_name += f"{fights_per_game}_fights_per_game_vs_{bot_behavior}"
    models_dir = os.path.join("models", experiment_name)
    logs_dir = os.path.join("logs", experiment_name)

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    if verbose :
        print(f"Saving directories created for {experiment_name} experiment")

    return logs_dir, models_dir