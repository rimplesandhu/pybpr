import os
from pybpr import RecSys, UserItemData, HybridMF
from pybpr import bpr_loss, hinge_loss, bpr_loss_v2
import torch
import pandas as pd
import numpy as np
from functools import partial
import multiprocessing as mp
import itertools


def build_recsys(
    rdf,
    tdf,
    run,
    neg_option,
    item_option,
    n_latent,
    learning_rate,
    loss_function,
    weight_decay,
    n_iter=500,
    batch_size=1000,
    eval_every=5,
    save_every=5,
    eval_user_size=70000,
    output_dir="/kfs2/projects/zazzle/pybpr/examples/output/movielens3/",
):
    """Build and train a recommendation system. """

    # Name the run
    name = f'{run}_{item_option}_{loss_function.__name__}'
    name += f'_{neg_option}'
    name += f'_ld{n_latent}_lr{int(learning_rate*1000)}'
    name += f'_wd{int(weight_decay*1000)}'
    print(f"Starting process: {name}", flush=True)

    # build data object
    ui = UserItemData(name=name)
    ui.add_positive_interactions(
        user_ids=rdf.UserID[rdf.Rating >= 4.0],
        item_ids=rdf.MovieID[rdf.Rating >= 4.0]
    )
    if neg_option != 'neg-ignore':
        ui.add_negative_interactions(
            user_ids=rdf.UserID[rdf.Rating < 4.0],
            item_ids=rdf.MovieID[rdf.Rating < 4.0]
        )
    ui.add_user_features(
        user_ids=rdf.UserID.unique(),
        feature_ids=rdf.UserID.unique()
    )
    if item_option == 'metadata':
        ui.add_item_features(
            item_ids=tdf.MovieID,
            feature_ids=tdf.TagID
        )
    elif item_option == 'indicator':
        ui.add_item_features(
            item_ids=tdf.MovieID.unique(),
            feature_ids=tdf.MovieID.unique()
        )
    elif item_option == 'both':
        ui.add_item_features(
            item_ids=np.concatenate(
                (tdf.MovieID.values, tdf.MovieID.unique())),
            feature_ids=np.concatenate(
                (tdf.TagID.values, tdf.TagID.max() + tdf.MovieID.unique()))
        )
    else:
        raise ValueError(f"Unknown item_features type: {item_option}")

    # Split data into train and test sets
    ui.train_test_split(
        train_ratio_pos=0.8,
        train_ratio_neg=0.0 if neg_option == 'neg-test' else 0.8,
        show_progress=False
    )
    print(ui, flush=True)

    # recomender
    recommender = RecSys(
        data=ui,
        model=HybridMF(ui.n_user_features,
                       ui.n_item_features, n_latent=n_latent),
        optimizer=partial(
            torch.optim.Adam, lr=learning_rate, weight_decay=weight_decay
        ),
        loss_function=loss_function,
        output_dir=os.path.join(output_dir, ui.name),
        log_level=1
    )
    # print(f"[{name}] {recommender}", flush=True)

    # Train the model
    recommender.fit(
        n_iter=n_iter,
        batch_size=batch_size,
        eval_every=eval_every,
        save_every=save_every,
        eval_user_size=eval_user_size,
        explicit_ns_for_train=True if neg_option == 'neg-both' else False,
        explicit_ns_for_test=False if neg_option == 'neg-ignore' else True,
        disable_progress_bar=True
    )

    print(f"Finished process: {name}")
    return recommender


def run_experiment(params, rdf, tdf):
    """Run a single experiment with the given parameters.  """
    try:
        recommender = build_recsys(rdf, tdf, **params)
        return f"Successfully completed: {recommender.data.name}"
    except Exception as e:
        return f"Failed: {str(e)}"


if __name__ == '__main__':
    # Enable multiprocessing for PyTorch (optional, may improve performance)
    torch.set_num_threads(1)  # Limit threads per process

    # Load raw data
    data_dir = '/home/rsandhu/zazzle/raw_data'
    ratings_path = os.path.join(data_dir, 'ml-10M100K', 'ratings.dat')
    tags_path = os.path.join(data_dir, 'tag-genome', 'tag_relevance.dat')

    # Read ratings data
    rdf = pd.read_csv(
        ratings_path,
        sep='::',
        engine='python',
        header=None
    )
    rdf.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

    # Read tag data
    tdf = pd.read_csv(tags_path, sep='\t', header=None)
    tdf.columns = ['MovieID', 'TagID', 'Relevance']
    tdf.drop(index=tdf.index[tdf.Relevance < 0.8], inplace=True)

    # Define parameter grid - without rdf and tdf
    param_grid = {
        'run': list(range(5)),
        'item_option': ['metadata', 'indicator', 'both'],
        'n_latent': [32, 64, 128],
        'learning_rate': [0.005, 0.01, 0.05],
        'loss_function': [bpr_loss, bpr_loss_v2, hinge_loss],
        'weight_decay': [0],
        'neg_option': ['neg-ignore', 'neg-test', 'neg-both'],
        # Add any other parameters you want to vary
    }

    # param_grid = {
    #     'run': list(range(1)),
    #     'item_option': ['metadata', 'indicator', 'both'],
    #     'n_latent': [64],
    #     'learning_rate': [0.005, 0.01],
    #     'loss_function': [bpr_loss, bpr_loss_v2],
    #     'weight_decay': [0],
    #     'neg_option': ['neg-ignore', 'neg-test', 'neg-both'],
    #     # Add any other parameters you want to vary
    # }

    # Generate all parameter combinations
    all_params = []
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    for values in itertools.product(*param_values):
        params = dict(zip(param_names, values))
        all_params.append(params)

    # Print experiment plan
    print(f"Running {len(all_params)} experiments")

    # Create a partial function with fixed rdf and tdf
    run_with_fixed_data = partial(run_experiment, rdf=rdf, tdf=tdf)

    # Set the number of processes to use
    num_processes = min(len(all_params), mp.cpu_count())
    print(f"Using {num_processes} processes")

    # Run experiments in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_with_fixed_data, all_params)

    # Print results summary
    print("\nExperiment Results:")
    for result in results:
        print(f"- {result}")
