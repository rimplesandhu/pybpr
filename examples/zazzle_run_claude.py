import os
from pybpr import RecSys, UserItemData, HybridMF
from pybpr import bpr_loss, bpr_loss_v2, hinge_loss
import torch
import pandas as pd
import numpy as np
from functools import partial
import multiprocessing as mp
import itertools
import matplotlib.pyplot as plt
import json
from typing import List, Tuple


def save_metrics_and_plots(recommender: RecSys, name: str) -> None:
    """Save training metrics and generate loss plots."""
    try:
        # Save metrics to JSON
        recommender.save_metrics("training_metrics.json")
        
        # Save model
        model_path = recommender.save_model("model.torch")
        print(f"Model saved to: {model_path}")
        
        # Generate loss plot if metrics are available
        if recommender.metrics:
            epochs = [m['epoch'] for m in recommender.metrics]
            losses = [m['loss'] for m in recommender.metrics]
            
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.plot(epochs, losses, 'b-', linewidth=2)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            
            # Plot AUC if available
            test_aucs = [m.get('test_auc', None) for m in recommender.metrics]
            test_aucs = [auc for auc in test_aucs if auc is not None]
            
            if test_aucs:
                auc_epochs = [m['epoch'] for m in recommender.metrics if 'test_auc' in m]
                plt.subplot(1, 2, 2)
                plt.plot(auc_epochs, test_aucs, 'r-', linewidth=2)
                plt.title('Test AUC')
                plt.xlabel('Epoch')
                plt.ylabel('AUC')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(recommender.output_dir, "training_curves.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Training curves saved to: {plot_path}")
            
            # Print final metrics
            final_metrics = recommender.metrics[-1]
            print(f"Final metrics for {name}:")
            print(f"  Final Loss: {final_metrics['loss']:.4f}")
            if 'test_auc' in final_metrics:
                print(f"  Final Test AUC: {final_metrics['test_auc']:.4f}")
            if 'test_precision_10' in final_metrics:
                print(f"  Test Precision@10: {final_metrics['test_precision_10']:.4f}")
            if 'test_recall_10' in final_metrics:
                print(f"  Test Recall@10: {final_metrics['test_recall_10']:.4f}")
                
    except Exception as e:
        print(f"Error saving metrics for {name}: {e}")


def generate_example_predictions(recommender: RecSys, pdf: pd.DataFrame, 
                                 idf: pd.DataFrame, name: str) -> None:
    """Generate example predictions and save them."""
    try:
        # Get some sample users and items for prediction
        sample_users = recommender.users[:min(5, len(recommender.users))]
        sample_items = list(range(min(10, recommender.data.n_items)))
        
        predictions_data = []
        
        for user_idx in sample_users:
            try:
                # Get predictions for this user across sample items
                scores = recommender.predict(users=[user_idx], items=sample_items)
                
                # Get top 5 items for this user
                top_indices = np.argsort(scores)[-5:][::-1]
                top_scores = scores[top_indices]
                
                user_predictions = {
                    'user_internal_idx': int(user_idx),
                    'top_items': []
                }
                
                for item_idx, score in zip(top_indices, top_scores):
                    # Try to get the original product ID
                    try:
                        product_id = recommender.data.get_id(item_idx, 'item')
                    except Exception as e:
                        print(f"Warning: Could not get product ID for item {item_idx}: {e}")
                        product_id = item_idx  # Fallback to internal index

                    user_predictions['top_items'].append({
                        'item_internal_idx': int(item_idx),
                        'product_id': int(product_id),
                        'predicted_score': float(score)
                    })
                
                predictions_data.append(user_predictions)
                
            except Exception as e:
                print(f"Error predicting for user {user_idx}: {e}")
                continue
        
        # Save predictions to JSON
        predictions_path = os.path.join(recommender.output_dir, "example_predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump({
                'model_name': name,
                'n_users_predicted': len(predictions_data),
                'n_items_evaluated': len(sample_items),
                'predictions': predictions_data
            }, f, indent=2)
        
        print(f"Example predictions saved to: {predictions_path}")
        
        # Print summary
        print(f"Generated predictions for {len(predictions_data)} users")
        if predictions_data:
            avg_top_score = np.mean([p['top_items'][0]['predicted_score'] 
                                   for p in predictions_data if p['top_items']])
            print(f"Average top prediction score: {avg_top_score:.4f}")
            
    except Exception as e:
        print(f"Error generating predictions for {name}: {e}")


def predict_for_test_users(recommender: RecSys, pdf: pd.DataFrame,
                          idf: pd.DataFrame, name: str, n_users: int = 10) -> None:
    """Generate predictions for test set users and show their positive/negative interactions."""
    try:
        print(f"Generating predictions for test set users: {name}")

        # Get test set users from positive interactions
        test_users = np.unique(recommender.data.Rpos_test.row)[:n_users]
        print(f"Found {len(test_users)} test users, predicting for first {n_users}")

        # Get all items for prediction
        all_items = list(range(recommender.data.n_items))

        predictions_data = {
            'model_name': name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_test_users': len(test_users),
            'users_predictions': []
        }

        for user_idx in test_users:
            try:
                # Get predictions for this user across all items
                scores = recommender.predict(users=[user_idx], items=all_items)

                # Get top 10 predictions
                top_indices = np.argsort(scores)[-10:][::-1]
                top_scores = scores[top_indices]

                # Get positive interactions for this user in test set
                pos_test_mask = recommender.data.Rpos_test.row == user_idx
                pos_test_items = recommender.data.Rpos_test.col[pos_test_mask].tolist()

                # Get negative interactions for this user in test set (if available)
                neg_test_items = []
                if recommender.data.Rneg_test is not None and recommender.data.Rneg_test.nnz > 0:
                    neg_test_mask = recommender.data.Rneg_test.row == user_idx
                    neg_test_items = recommender.data.Rneg_test.col[neg_test_mask].tolist()

                # Get positive interactions from training set too for context
                pos_train_mask = recommender.data.Rpos_train.row == user_idx
                pos_train_items = recommender.data.Rpos_train.col[pos_train_mask].tolist()

                # Convert internal indices to original product IDs
                try:
                    pos_test_product_ids = [recommender.data.get_id(item_idx, 'item') for item_idx in pos_test_items]
                    pos_train_product_ids = [recommender.data.get_id(item_idx, 'item') for item_idx in pos_train_items]
                    neg_test_product_ids = [recommender.data.get_id(item_idx, 'item') for item_idx in neg_test_items]
                    top_predicted_product_ids = [recommender.data.get_id(item_idx, 'item') for item_idx in top_indices]

                    # Get original user ID
                    user_id = recommender.data.get_id(user_idx, 'user')

                except Exception as e:
                    print(f"Warning: Could not convert indices to IDs for user {user_idx}: {e}")
                    pos_test_product_ids = pos_test_items
                    pos_train_product_ids = pos_train_items
                    neg_test_product_ids = neg_test_items
                    top_predicted_product_ids = top_indices.tolist()
                    user_id = user_idx

                user_prediction = {
                    'user_internal_idx': int(user_idx),
                    'user_original_id': int(user_id),
                    'positive_test_items': pos_test_product_ids,
                    'positive_train_items': pos_train_product_ids[:10],  # Show only first 10 for brevity
                    'negative_test_items': neg_test_product_ids,
                    'top_10_predictions': []
                }

                # Add top predictions with scores
                for item_idx, score in zip(top_predicted_product_ids, top_scores):
                    user_prediction['top_10_predictions'].append({
                        'predicted_product_id': int(item_idx),
                        'predicted_score': float(score),
                        'in_test_positive': int(item_idx) in pos_test_product_ids,
                        'in_train_positive': int(item_idx) in pos_train_product_ids,
                        'in_test_negative': int(item_idx) in neg_test_product_ids
                    })

                predictions_data['users_predictions'].append(user_prediction)

                # Print summary for this user
                print(f"User {user_id}: {len(pos_test_product_ids)} pos test, "
                      f"{len(pos_train_product_ids)} pos train, "
                      f"{len(neg_test_product_ids)} neg test items")

                # Check how many predictions match positive test items
                matches = sum(1 for pred in user_prediction['top_10_predictions']
                             if pred['in_test_positive'])
                if matches > 0:
                    print(f"  -> {matches}/10 predictions match test positive items!")

            except Exception as e:
                print(f"Error predicting for user {user_idx}: {e}")
                continue

        # Save predictions to JSON
        predictions_path = os.path.join(recommender.output_dir, "test_user_predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)

        print(f"Test user predictions saved to: {predictions_path}")

        # Print summary statistics
        if predictions_data['users_predictions']:
            total_matches = sum(
                sum(1 for pred in user_pred['top_10_predictions'] if pred['in_test_positive'])
                for user_pred in predictions_data['users_predictions']
            )
            total_predictions = len(predictions_data['users_predictions']) * 10
            hit_rate = total_matches / total_predictions if total_predictions > 0 else 0

            print(f"Summary: {total_matches}/{total_predictions} predictions hit test positive items")
            print(f"Hit rate: {hit_rate:.4f}")

            # Show average scores for hits vs misses
            hit_scores = []
            miss_scores = []
            for user_pred in predictions_data['users_predictions']:
                for pred in user_pred['top_10_predictions']:
                    if pred['in_test_positive']:
                        hit_scores.append(pred['predicted_score'])
                    else:
                        miss_scores.append(pred['predicted_score'])

            if hit_scores:
                print(f"Average score for hits: {np.mean(hit_scores):.4f}")
            if miss_scores:
                print(f"Average score for misses: {np.mean(miss_scores):.4f}")

    except Exception as e:
        print(f"Error generating test predictions for {name}: {e}")


def create_experiment_summary(successful_runs: List[dict]) -> None:
    """Create a summary report of all successful experiments."""
    try:
        summary_data = {
            'experiment_summary': {
                'total_successful_runs': len(successful_runs),
                'timestamp': pd.Timestamp.now().isoformat(),
                'runs': []
            }
        }
        
        for result in successful_runs:
            recommender = result['recommender']
            run_summary = {
                'name': result['name'],
                'model_config': {
                    'n_latent': recommender.model.n_latent,
                    'n_user_features': recommender.data.n_user_features,
                    'n_item_features': recommender.data.n_item_features,
                    'n_users': len(recommender.users),
                    'n_items': recommender.data.n_items
                },
                'final_metrics': {}
            }
            
            # Add final metrics if available
            if recommender.metrics:
                final_metrics = recommender.metrics[-1]
                run_summary['final_metrics'] = {
                    'epoch': final_metrics['epoch'],
                    'loss': final_metrics['loss']
                }
                for key in ['test_auc', 'test_precision_10', 'test_recall_10', 'test_ndcg_10']:
                    if key in final_metrics:
                        run_summary['final_metrics'][key] = final_metrics[key]
            
            summary_data['experiment_summary']['runs'].append(run_summary)
        
        # Save summary
        summary_path = "output/zazzle/experiment_summary.json"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Experiment summary saved to: {summary_path}")
        
        # Print best performing model
        if successful_runs and successful_runs[0]['recommender'].metrics:
            best_auc = 0
            best_run = None
            for result in successful_runs:
                if result['recommender'].metrics:
                    final_metrics = result['recommender'].metrics[-1]
                    test_auc = final_metrics.get('test_auc', 0)
                    if test_auc > best_auc:
                        best_auc = test_auc
                        best_run = result
            
            if best_run:
                print(f"\nBest performing model: {best_run['name']}")
                print(f"Best Test AUC: {best_auc:.4f}")
        
    except Exception as e:
        print(f"Error creating experiment summary: {e}")


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
    output_dir="output/zazzle/",
):
    """Build and train a recommendation system. """

    # Name the run
    name = f'{run}_{item_option}_{loss_function.__name__}'
    name += f'_{neg_option}'
    name += f'_ld{n_latent}_lr{int(learning_rate*1000)}'
    name += f'_wd{int(weight_decay*1000)}'
    print(f"Starting process: {name}", flush=True)

    print(f"DEBUG: About to create UserItemData", flush=True)
    ui = UserItemData(name=name)
    print(f"DEBUG: Created UserItemData", flush=True)

    # Add positive interactions (ratings >= 4.0)
    ui.add_positive_interactions(
        user_ids=pdf.user_member_id,
        item_ids=pdf.product_id
    )
    if neg_option != 'neg-ignore':
        ui.add_negative_interactions(
            user_ids=ndf.user_member_id,
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

    print(f"DEBUG: About to create RecSys for {name}", flush=True)
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

    print(f"DEBUG: About to start training for {name}", flush=True)
    # Train the model
    recommender.fit(
        n_iter=n_iter,
        batch_size=batch_size,
        eval_every=eval_every,
        #save_every=save_every,
        eval_user_size=eval_user_size,
        #explicit_ns_for_train=True if neg_option == 'neg-both' else False,
        #explicit_ns_for_test=False if neg_option == 'neg-ignore' else True,
        #disable_progress_bar=True
    )
    print(f"Finished process: {name}")
    
    # Save metrics and generate plots
    print(f"Saving metrics and plots for {name}", flush=True)
    save_metrics_and_plots(recommender, name)
    
    # Generate example predictions
    print(f"Generating example predictions for {name}", flush=True)
    generate_example_predictions(recommender, pdf, idf, name)

    # Generate predictions for test users
    print(f"Generating test user predictions for {name}", flush=True)
    predict_for_test_users(recommender, pdf, idf, name, n_users=10)

    print(f"Finished process: {name}")
    return recommender


def run_experiment(params, pdf, ndf, idf):
    """Run a single experiment with the given parameters.  """
    try:
        recommender = build_recsys(pdf, ndf, idf, **params)
        return {
            'status': 'success',
            'name': recommender.data.name,
            'recommender': recommender,
            'message': f"Successfully completed: {recommender.data.name}"
        }
    except Exception as e:
        return {
            'status': 'failed',
            'name': f"run_{params.get('run', 'unknown')}",
            'error': str(e),
            'message': f"Failed: {str(e)}"
        }


if __name__ == '__main__':
    # Enable multiprocessing for PyTorch (optional, may improve performance)
    torch.set_num_threads(1)  # Limit threads per process

    # Load raw data
    data_dir = '../data/NREL2'

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
            subset=['user_member_id', 'product_id'], keep='last', inplace=True)

    # Define parameter grid - without rdf and tdf
    param_grid = {
        'run': list(range(1)),
        'item_option': ['metadata'],#, 'indicator'],
        'n_latent': [64],
        'learning_rate': [0.05],
        'loss_function': [bpr_loss],
        'weight_decay': [0],
        'neg_option': ['neg-ignore']#, 'neg-test'],
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

    # Run experiments sequentially to avoid matplotlib/multiprocessing issues
    print("Running experiments sequentially to support plotting...")
    results = []
    for i, params in enumerate(all_params):
        print(f"\nRunning experiment {i+1}/{len(all_params)}: {params}")
        result = run_with_fixed_data(params)
        results.append(result)

        # Print immediate result
        print(f"Result: {result['message']}")

    # Print results summary
    print("\nExperiment Results Summary:")
    successful_runs = [r for r in results if r['status'] == 'success']
    failed_runs = [r for r in results if r['status'] == 'failed']

    print(f"Successful runs: {len(successful_runs)}")
    print(f"Failed runs: {len(failed_runs)}")

    for result in successful_runs:
        print(f"✓ {result['message']}")

    for result in failed_runs:
        print(f"✗ {result['message']}")

    # Create summary report
    if successful_runs:
        print("\nCreating summary report...")
        create_experiment_summary(successful_runs)
