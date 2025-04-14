
"""
Example usage of the improved RecSys class with recommendations.
"""

import os
import json
from pybpr import RecSys, UserItemData, HybridMF
import torch
import numpy as np
import pandas as pd
import logging
from functools import partial


# Set up logging
# logging.basicConfig(level=logging.INFO)

# Load your data
data_dir = '/home/rsandhu/zazzle/raw_data'
rdf = pd.read_csv(os.path.join(data_dir, 'ml-10M100K',
                  'ratings.dat'), sep='::', engine='python', header=None)
rdf.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
print(rdf.head())

tdf = pd.read_csv(os.path.join(data_dir, 'tag-genome',
                  'tag_relevance.dat'), sep='\t', header=None)
tdf.columns = ['MovieID', 'TagID', 'Relevance']
tdf.drop(index=tdf.index[tdf.Relevance < 0.8], inplace=True)
print(tdf.head())

# define data
ui = UserItemData(name='User-Metadata-only')
ui.add_positive_interactions(
    user_ids=rdf.UserID[rdf.Rating >= 4.],
    item_ids=rdf.MovieID[rdf.Rating >= 4.]
)
ui.add_negative_interactions(
    user_ids=rdf.UserID[rdf.Rating < 4.],
    item_ids=rdf.MovieID[rdf.Rating < 4.]
)
ui.add_user_features(
    user_ids=rdf.UserID.unique(),
    feature_ids=rdf.UserID.unique()
)
ui.add_item_features(
    item_ids=tdf.MovieID,
    feature_ids=tdf.TagID
)
ui.train_test_split(
    train_ratio_pos=0.8,
    train_ratio_neg=0.8,
    show_progress=True
)
print(ui)

# Initialize recommendation system and fit
recommender = RecSys(
    ui_data=ui,
    n_latent=64,
    optimizer=partial(torch.optim.Adam, lr=0.01, weight_decay=0.),
    output_dir='/kfs2/projects/zazzle/pybpr/notebooks/output/ml-run',
    log_level=1
)
recommender.fit(
    n_iter=500,
    batch_size=2000,
    eval_every=5,
    eval_sample_size=10000,
    disable_progress_bar=False
)
recommender.save_model()

# Plot the training metrics
# recommender.plot_metric(metric="test_auc_mean", show_plot=True)

# Save the trained model


# # Get recommendations for a specific user
# user_id = 42  # Replace with an actual user ID
# top_recommendations = recommender.get_top_k_recommendations(
#     user_idx=user_id,
#     k=10,
#     exclude_known=True
# )

# # Print the recommendations
# print(f"Top 10 recommendations for user {user_id}:")
# for i, (item_id, score) in enumerate(top_recommendations, 1):
#     print(f"{i}. Item {item_id} (Score: {score:.4f})")

# # Evaluate on specific users
# power_users = [42, 100, 200]  # Replace with actual user IDs
# user_metrics = recommender.evaluate_specific_users(user_ids=power_users)
# print(f"Metrics for specific users: {user_metrics}")

# # Export recommendations for all users
# all_users = recommender.users_test[:100]  # Get first 100 test users
# all_recommendations = {}

# for user_id in all_users:
#     user_recs = recommender.get_top_k_recommendations(
#         user_idx=user_id,
#         k=5,
#         exclude_known=True
#     )
#     all_recommendations[user_id] = [item_id for item_id, _ in user_recs]

# # Save recommendations to file
# with open("./recommendation_output/user_recommendations.json", "w") as f:
#     json.dump(all_recommendations, f)

# print("Recommendations saved to file.")
