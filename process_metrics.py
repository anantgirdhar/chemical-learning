from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
import pickle
import torch
from tqdm import tqdm

##### FILE IO #####

def parse_metrics(epochs, metrics):
    model_data = {
            'epoch': range(1, epochs+1),
            'loss_train': metrics['train_loss'],
            'loss_test': metrics['test_loss'],
            'PC1_train': [r[0] for r in metrics['train_PC']],
            'PC2_train': [r[1] for r in metrics['train_PC']],
            'PC3_train': [r[2] for r in metrics['train_PC']],
            'PC1_test': [r[0] for r in metrics['test_PC']],
            'PC2_test': [r[1] for r in metrics['test_PC']],
            'PC3_test': [r[2] for r in metrics['test_PC']],
            'BFL1_train': [r[0][1] for r in metrics['train_BFL']],
            'BFL2_train': [r[1][1] for r in metrics['train_BFL']],
            'BFL3_train': [r[2][1] for r in metrics['train_BFL']],
            'BFL1_test': [r[0][1] for r in metrics['test_BFL']],
            'BFL2_test': [r[1][1] for r in metrics['test_BFL']],
            'BFL3_test': [r[2][1] for r in metrics['test_BFL']],
            }
    return model_data

def read_model_file(filename):
    checkpoint = torch.load(filename)
    epochs = checkpoint['epochs']
    metrics = checkpoint['metrics']
    return parse_metrics(epochs, metrics)

def read_folds_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    epochs = data['epochs']
    metrics = data['metrics']
    return {k: parse_metrics(epochs, m) for k, m in metrics.items()}

def get_params_from_filename(filename, include_layers=False):
    filename = os.path.basename(filename)
    tokens = filename.split('.')[0].split('_')
    if 'layers' in filename:
        # This is a non-uniform model
        model_metadata = {
                'inputs': int(tokens[1][len('in'):]),
                'outputs': int(tokens[2][len('out'):]),
                'dropout': True if 'dropout' in tokens else False,
                'order': False if 'unorderedsp' in tokens else True,
                'atom_counts': True if 'withcounts' in tokens else False,
                'uniform': False,
                }
        if include_layers:
            model_metadata['layers'] = [
                    int(size)
                    for size in tokens[3][len('layers'):].split('-')
                    ],
    else:
        # This is a uniform model
        model_metadata = {
                'inputs': int(tokens[1][len('in'):]),
                'outputs': int(tokens[2][len('out'):]),
                'num_layers': int(tokens[3][len('nl'):]),
                'npl': int(tokens[4][len('npl'):]),
                'dropout': True if 'dropout' in tokens else False,
                'order': False if 'unorderedsp' in tokens else True,
                'atom_counts': True if 'withcounts' in tokens else False,
                'uniform': True,
                }
    return model_metadata

def compile_model_files(filename_list):
    compiled_df = None
    for filename in filename_list:
        model_data = read_model_file(filename)
        model_metadata = get_params_from_filename(filename)
        df = pd.DataFrame(model_data)
        metadata_keys = list(model_metadata.keys())
        metadata_values = [model_metadata[k] for k in metadata_keys]
        df[metadata_keys] = metadata_values
        if compiled_df is None:
            compiled_df = df
        else:
            compiled_df = pd.concat([compiled_df, df], ignore_index=True)
    return compiled_df

def compile_fold_files(filename_list):
    compiled_df = None
    pbar = tqdm(filename_list)
    for filename in pbar:
        pbar.set_description(f'{os.path.basename(filename):<80s}')
        model_data = read_folds_file(filename)
        model_metadata = get_params_from_filename(filename)
        for fold, metrics in model_data.items():
            df = pd.DataFrame(metrics)
            metadata_keys = list(model_metadata.keys())
            metadata_values = [model_metadata[k] for k in metadata_keys]
            df[metadata_keys] = metadata_values
            df['fold'] = int(fold[len('fold'):])
            df['model_name'] = os.path.basename(filename)[:len(filename)-len('_folds.p')]
            if compiled_df is None:
                compiled_df = df
            else:
                compiled_df = pd.concat([compiled_df, df], ignore_index=True)
    return compiled_df

def get_compiled_df(compiled_data_filename='compiled_data.p'):
    if not os.path.isfile(compiled_data_filename):
        df = compile_files(glob('output_models/*.pth'))
        df.to_pickle(compiled_data_filename)
    else:
        df = pd.read_pickle(compiled_data_filename)
    df['PC_avg_train'] = df[['PC1_train', 'PC2_train', 'PC3_train']].mean(axis=1)
    df['PC_avg_test'] = df[['PC1_test', 'PC2_test', 'PC3_test']].mean(axis=1)
    df['PC_var_train'] = df[['PC1_train', 'PC2_train', 'PC3_train']].var(axis=1)
    df['PC_var_test'] = df[['PC1_test', 'PC2_test', 'PC3_test']].var(axis=1)
    df['BFL_avg_train'] = df[['BFL1_train', 'BFL2_train', 'BFL3_train']].mean(axis=1)
    df['BFL_avg_test'] = df[['BFL1_test', 'BFL2_test', 'BFL3_test']].mean(axis=1)
    df['BFL_var_train'] = df[['BFL1_train', 'BFL2_train', 'BFL3_train']].var(axis=1)
    df['BFL_var_test'] = df[['BFL1_test', 'BFL2_test', 'BFL3_test']].var(axis=1)
    df['max_var'] = df[['PC_var_train', 'PC_var_test', 'BFL_var_train', 'BFL_var_test']].max(axis=1)
    return df

def get_compiled_folds_df(compiled_data_filename='compiled_folds_data.p'):
    if not os.path.isfile(compiled_data_filename):
        df = compile_fold_files(glob('cross_validation_pickles/*.p'))
        df.to_pickle(compiled_data_filename)
    else:
        df = pd.read_pickle(compiled_data_filename)
    df['PC_avg_train'] = df[['PC1_train', 'PC2_train', 'PC3_train']].mean(axis=1)
    df['PC_avg_test'] = df[['PC1_test', 'PC2_test', 'PC3_test']].mean(axis=1)
    df['PC_var_train'] = df[['PC1_train', 'PC2_train', 'PC3_train']].var(axis=1)
    df['PC_var_test'] = df[['PC1_test', 'PC2_test', 'PC3_test']].var(axis=1)
    df['BFL_avg_train'] = df[['BFL1_train', 'BFL2_train', 'BFL3_train']].mean(axis=1)
    df['BFL_avg_test'] = df[['BFL1_test', 'BFL2_test', 'BFL3_test']].mean(axis=1)
    df['BFL_var_train'] = df[['BFL1_train', 'BFL2_train', 'BFL3_train']].var(axis=1)
    df['BFL_var_test'] = df[['BFL1_test', 'BFL2_test', 'BFL3_test']].var(axis=1)
    df['max_var'] = df[['PC_var_train', 'PC_var_test', 'BFL_var_train', 'BFL_var_test']].max(axis=1)
    return df

##### DATA ANALYSIS #####

def get_mean_and_variance_across_folds(df):
    if df['epoch'].min() != df['epoch'].max():
        raise ValueError('The values should be computed for the same epoch.')
    groups = df[[
        'model_name',
        'epoch',
        'loss_train', 'loss_test',
        'PC1_train', 'PC1_test',
        'PC2_train', 'PC2_test',
        'PC3_train', 'PC3_test',
        'BFL1_train', 'BFL1_test',
        'BFL2_train', 'BFL2_test',
        'BFL3_train', 'BFL3_test',
        'inputs', 'outputs', 'dropout', 'order', 'atom_counts',
        'uniform', 'num_layers', 'npl'
        ]].groupby('model_name')
    mean = groups.mean()
    var = groups.var()
    # Rename the columsn so that the dataframes can be joined together
    no_rename_list = [
            'epoch',
            'inputs', 'outputs', 'dropout', 'order', 'atom_counts',
            'uniform', 'num_layers', 'npl',
            ]
    df_grouped = mean.merge(var, on='model_name', suffixes=('_mean', '_var'))
    df_grouped.drop(columns=[c + '_var' for c in no_rename_list], inplace=True)
    df_grouped.rename(columns={c+'_mean': c for c in no_rename_list}, inplace=True)
    # Compute some overall statistics
    for v in ['PC', 'BFL']:
        for t in ['train', 'test']:
            df_grouped[f'mean_{v}_{t}_mean'] = df_grouped[
                    [f'{v}{i}_{t}_mean' for i in range(1, 4)]
                    ].mean(axis=1, skipna=False)
            df_grouped[f'var_{v}_{t}_mean'] = df_grouped[
                    [f'{v}{i}_{t}_mean' for i in range(1, 4)]
                    ].var(axis=1, skipna=False)
            df_grouped[f'max_{v}_{t}_var'] = df_grouped[
                    [f'{v}{i}_{t}_var' for i in range(1, 4)]
                    ].max(axis=1, skipna=False)
            df_grouped[f'mean_{v}_{t}_var'] = df_grouped[
                    [f'{v}{i}_{t}_var' for i in range(1, 4)]
                    ].mean(axis=1, skipna=False)
    # Add in a couple of distance metrics
    # - distance of (PC1, PC2, PC3) from (1, 1, 1)
    # - distance of (BFL1, BFL2, BFL3) from (1, 1, 1)
    df_grouped['dist_PC_train'] = df_grouped[[
        'PC1_train_mean', 'PC2_train_mean', 'PC3_train_mean',
        ]].pow(2).sum(axis=1, skipna=False).pow(0.5)
    df_grouped['dist_PC_test'] = df_grouped[[
        'PC1_test_mean', 'PC2_test_mean', 'PC3_test_mean',
        ]].pow(2).sum(axis=1, skipna=False).pow(0.5)
    df_grouped['dist_BFL_train'] = df_grouped[[
        'BFL1_train_mean', 'BFL2_train_mean', 'BFL3_train_mean',
        ]].pow(2).sum(axis=1, skipna=False).pow(0.5)
    df_grouped['dist_BFL_test'] = df_grouped[[
        'BFL1_test_mean', 'BFL2_test_mean', 'BFL3_test_mean',
        ]].pow(2).sum(axis=1, skipna=False).pow(0.5)
    return df_grouped

def extract_best_models(final_df, BFL_tolerance=0.6, PC_tolerance=0.6):
    final_df = final_df.drop(
            final_df[
                (abs(final_df['BFL1_test']) < BFL_tolerance)
                | (abs(final_df['BFL2_test']) < BFL_tolerance)
                | (abs(final_df['BFL3_test']) < BFL_tolerance)
                | (abs(final_df['BFL1_train']) < BFL_tolerance)
                | (abs(final_df['BFL2_train']) < BFL_tolerance)
                | (abs(final_df['BFL3_train']) < BFL_tolerance)
                ].index
            )
    # As a first pass, we care about the average PC and BFL values
    # In the ideal case, all PC and BFL values will be 1
    # So sort the leftover data based on its distance from (1, 1, 1, 1)
    distance = [1, 1, 1, 1] - final_df[[
        'BFL_avg_train', 'BFL_avg_test', 'PC_avg_train', 'PC_avg_test'
        ]]
    distance = (distance ** 2.).sum(axis=1).pow(0.5)
    final_df['distance'] = distance
    return final_df.sort_values('distance', ascending=True)

def extract_best_crossvalidation_results(df):
    # Get top 5 models by each of the following criteria
    # - lowest dist_PC_train
    # - lowest dist_PC_test
    # - lowest dist_BFL_train
    # - lowest dist_BFL_test
    # - lowest max_PC_train_var
    # - lowest max_PC_test_var
    # - lowest max_BFL_train_var
    # - lowest max_BFL_test_var
    results = [
            df.sort_values('dist_PC_train').head(),
            df.sort_values('dist_PC_test').head(),
            df.sort_values('dist_BFL_train').head(),
            df.sort_values('dist_BFL_test').head(),
            df.sort_values('max_PC_train_var').head(),
            df.sort_values('max_PC_test_var').head(),
            df.sort_values('max_BFL_train_var').head(),
            df.sort_values('max_BFL_test_var').head(),
            df.sort_values('mean_PC_train_mean', ascending=False).head(),
            df.sort_values('mean_PC_test_mean', ascending=False).head(),
            df.sort_values('mean_BFL_train_mean', ascending=False).head(),
            df.sort_values('mean_BFL_test_mean', ascending=False).head(),
            ]
    results = pd.concat(results).drop_duplicates()
    # Sort the results by the max of the distances for each variable
    results['max_dist'] = df[[
        'dist_PC_train', 'dist_PC_test',
        'dist_BFL_train','dist_BFL_test',
        ]].max(axis=1, skipna=False)
    results = results.sort_values('max_dist')
    return results

##### PLOTTING CODE #####

def plot_3D_PC(df, colorby='max_var', cross_validation=False, colormap='afmhot'):
    if isinstance(colorby, str):
        colorby = df[colorby]
    fig = plt.figure()
    ax = Axes3D(fig)
    if cross_validation:
        PC1_label = 'PC1_test_mean'
        PC2_label = 'PC2_test_mean'
        PC3_label = 'PC3_test_mean'
    else:
        PC1_label = 'PC1_test'
        PC2_label = 'PC2_test'
        PC3_label = 'PC3_test'
    ax.scatter(
            df[PC1_label], df[PC2_label], df[PC3_label],
            c=colorby,
            cmap=colormap,
            )
    ax.set_clim = (colorby.min(), colorby.max())
    ax.set_xlabel(PC1_label)
    ax.set_ylabel(PC2_label)
    ax.set_zlabel(PC3_label)
    plt.show()

def plot_2D_PC(df, colorby='max_var'):
    if isinstance(colorby, str):
        colorby = df[colorby]
    # Drop the NAs from df and colorby
    drop_indices = colorby.isna()
    # Now plot
    clim = (colorby.min(), colorby.max())
    plt.subplots(1, 3)
    plt.subplot(131)
    plt.scatter(df['PC1_test'], df['PC2_test'], c=colorby, cmap='afmhot')
    plt.clim(clim)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.subplot(132)
    plt.scatter(df['PC1_test'], df['PC3_test'], c=colorby, cmap='afmhot')
    plt.clim(clim)
    plt.xlabel('PC1')
    plt.ylabel('PC3')
    plt.subplot(133)
    plt.scatter(df['PC2_test'], df['PC3_test'], c=colorby, cmap='afmhot')
    plt.clim(clim)
    plt.xlabel('PC2')
    plt.ylabel('PC3')
    plt.show()

def plot_PC_BFL_mean_var(df, colorby=(None, None)):
    plt.subplots(1, 2, sharex=True, sharey=True)
    plt.subplot(221)
    plt.scatter(df['mean_PC_train_mean'], df['max_PC_train_var'], c=colorby[0], cmap='afmhot_r')
    plt.title('PC_train')
    plt.ylabel('max of var')
    plt.subplot(223)
    plt.scatter(df['mean_PC_test_mean'], df['max_PC_test_var'], c=colorby[0], cmap='afmhot_r')
    plt.title('PC_test')
    plt.xlabel('mean of mean')
    plt.ylabel('max of var')
    plt.subplot(222)
    plt.scatter(df['mean_BFL_train_mean'], df['max_BFL_train_var'], c=colorby[1], cmap='afmhot_r')
    plt.title('BFL_train')
    plt.subplot(224)
    plt.scatter(df['mean_BFL_test_mean'], df['max_BFL_test_var'], c=colorby[1], cmap='afmhot_r')
    plt.title('BFL_test')
    plt.xlabel('mean of mean')
    plt.tight_layout()
    plt.show()

def plot_distribution(sequence, bins):
    mean = sequence.mean()
    median = sequence.median()
    plt.figure()
    plt.hist(sequence, bins=bins)
    plt.axvline(mean, color='red', linestyle='-', label='mean')
    plt.axvline(median, color='black', linestyle='--', label='median')
    plt.legend(loc='best')
    plt.show()

##### MAIN #####

def main_model():
    df = get_compiled_df()
    df2 = df[df['epoch'] == 20]
    return df, df2

def main_folds():
    df = get_compiled_folds_df()
    df2 = df[df['epoch'] == 10]
    df_grouped = get_mean_and_variance_across_folds(df2)
    best = extract_best_crossvalidation_results(df_grouped)
    with open('best_crossvalidation_models.p', 'wb') as f:
        pickle.dump(best, f)
    return df, df2, best

if __name__ == "__main__":
    main_folds()
