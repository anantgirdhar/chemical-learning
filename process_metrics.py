from glob import glob
import os
import pandas as pd
import torch

def read_file(filename):
    checkpoint = torch.load(filename)
    epochs = checkpoint['epochs']
    metrics = checkpoint['metrics']
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

def get_params_from_filename(filename):
    filename = os.path.basename(filename)
    tokens = filename.split('.')[0].split('_')
    if 'layers' in filename:
        # This is a non-uniform model
        model_metadata = {
                'inputs': int(tokens[1][len('in'):]),
                'outputs': int(tokens[2][len('out'):]),
                # 'layers': [
                #     int(size)
                #     for size in tokens[3][len('layers'):].split('-')
                #     ],
                'dropout': True if 'dropout' in tokens else False,
                'order': False if 'unorderedsp' in tokens else True,
                'atom_counts': True if 'withcounts' in tokens else False,
                'uniform': False,
                }
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

def compile_files(filename_list):
    compiled_df = None
    for filename in filename_list:
        model_data = read_file(filename)
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

def get_df(compiled_data_filename='compiled_data.p'):
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

def main():
    df = get_df()
    df2 = df[df['epoch'] == 20]
    return df, df2
