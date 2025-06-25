import os
from pybpr import RecSys, UserItemData, HybridMF
from pybpr import bpr_loss, bpr_loss_v2, hinge_loss
import torch
import pandas as pd
import numpy as np
from functools import partial
import multiprocessing as mp
import itertools


def build_recsys(
    pdf,
    ndf,
    idf,
    run,
    neg_option,
    item_option,
    n_latent,
    learning_rate,
    loss_function,
    weight_decay,
    n_iter=10,
    batch_size=1000,
    eval_every=10,
    save_every=10,
    eval_user_size=10000,
    output_dir="/kfs2/projects/zazzle/pybpr/examples/output/zazzle/",
):
    """Build and train a recommendation system. """

    # Name the run
    name = f'{run}_{item_option}_{loss_function.__name__}'
    name += f'_{neg_option}'
    name += f'_ld{n_latent}_lr{int(learning_rate*1000)}'
    name += f'_wd{int(weight_decay*1000)}'
    print(f"Starting process: {name}", flush=True)

    print(f"Starting process: {name}")
    ui = UserItemData(name=name)

    # Add positive interactions (ratings >= 4.0)
    ui.add_positive_interactions(
        user_ids=pdf.user_id,
        item_ids=pdf.product_id
    )
    if neg_option != 'neg-ignore':
        ui.add_negative_interactions(
            user_ids=ndf.user_id,
            item_ids=ndf.product_id
        )
    ui.add_user_features(
        user_ids=ui.user_ids_in_interactions,
        feature_ids=ui.user_ids_in_interactions
    )
    idf = idf[idf.product_id.isin(ui.item_ids_in_interactions)]
    if item_option == 'metadata':
        ui.add_item_features(
            item_ids=idf.product_id,
            feature_ids=idf.final_department_id
        )
    elif item_option == 'indicator':
        ui.add_item_features(
            item_ids=ui.item_ids_in_interactions,
            feature_ids=ui.item_ids_in_interactions
        )
    else:
        raise ValueError(f"Unknown item_features type: {item_option}")

    # Split data into train and test sets
    ui.train_test_split(
        train_ratio_pos=0.8,
        train_ratio_neg=0.0 if neg_option == 'neg-test' else 0.8,
        show_progress=False
    )

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

    # print(f"[{name}] {recommender}")

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


def run_experiment(params, pdf, ndf, idf):
    """Run a single experiment with the given parameters.  """
    try:
        recommender = build_recsys(pdf, ndf, idf, **params)
        return f"Successfully completed: {recommender.data.name}"
    except Exception as e:
        return f"Failed: {str(e)}"


if __name__ == '__main__':
    # Enable multiprocessing for PyTorch (optional, may improve performance)
    torch.set_num_threads(1)  # Limit threads per process

    # Load raw data
    data_dir = '/kfs2/projects/zazzle/raw_data/NREL'

    # viewed
    files = [os.path.join(
        data_dir, f'Clicks_{str(ix).zfill(4)}_part_00.parquet') for ix in range(80)]
    df_viewed = pd.concat(
        [pd.read_parquet(ifile, engine='fastparquet') for ifile in files])

    # ordered
    files = [os.path.join(
        data_dir, f'OrderItems_{str(ix).zfill(4)}_part_00.parquet') for ix in range(80)]
    df_ordered = pd.concat([pd.read_parquet(ifile) for ifile in files])

    # item info
    files = [os.path.join(
        data_dir, f'Products_{str(ix).zfill(4)}_part_00.parquet') for ix in range(80)]
    df_item_info = pd.concat([pd.read_parquet(ifile) for ifile in files])

    # manage
    df_clicked_not_ordered = df_viewed[df_viewed['is_click']].copy()
    df_viewed_not_clicked = df_viewed[~df_viewed['is_click']].copy()
    df_clicked = pd.concat([df_ordered, df_clicked_not_ordered])

    # prune
    df_list = [df_viewed, df_clicked, df_ordered,
               df_viewed_not_clicked, df_clicked_not_ordered]
    for idf in df_list:
        idf.drop_duplicates(
            subset=['user_id', 'product_id'], keep='last', inplace=True)

    # Define parameter grid - without rdf and tdf
    param_grid = {
        'run': list(range(1)),
        'item_option': ['metadata', 'indicator'],
        'n_latent': [64],
        'learning_rate': [0.05],
        'loss_function': [bpr_loss],
        'weight_decay': [0],
        'neg_option': ['neg-ignore', 'neg-test'],
        # Add any other parameters you want to vary
    }

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
    run_with_fixed_data = partial(
        run_experiment,
        pdf=df_ordered,
        ndf=df_clicked_not_ordered,
        idf=df_item_info
    )

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
