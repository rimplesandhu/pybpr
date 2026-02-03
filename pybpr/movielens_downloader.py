#!/usr/bin/env python3
"""
Direct download of MovieLens datasets with tags support.
Includes ratings, movies, tags, and tag genome data.
"""

import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from .utils import get_logger


class MovieLensDownloader:
    """Download and load MovieLens datasets with tag support."""

    BASE_URL = "https://files.grouplens.org/datasets/movielens/"

    DATASETS = {
        'ml-100k': {
            'url': 'ml-100k.zip',
            'extract_dir': 'ml-100k',
            'size': '5MB',
            'ratings': 100000,
            'has_tags': False,
            'file_mappings': {
                'ratings': 'u.data',
                'movies': 'u.item',
                'users': 'u.user'
            }
        },
        'ml-1m': {
            'url': 'ml-1m.zip',
            'extract_dir': 'ml-1m',
            'size': '6MB',
            'ratings': 1000000,
            'has_tags': False,
            'file_mappings': {
                'ratings': 'ratings.dat',
                'movies': 'movies.dat',
                'users': 'users.dat'
            }
        },
        'ml-10m': {
            'url': 'ml-10m.zip',
            'extract_dir': 'ml-10M100K',
            'size': '63MB',
            'ratings': 10000000,
            'has_tags': True,
            'file_mappings': {
                'ratings': 'ratings.dat',
                'movies': 'movies.dat',
                'tags': 'tags.dat'
            }
        },
        'ml-20m': {
            'url': 'ml-20m.zip',
            'extract_dir': 'ml-20m',
            'size': '190MB',
            'ratings': 20000000,
            'has_tags': True,
            'file_mappings': {
                'ratings': 'ratings.csv',
                'movies': 'movies.csv',
                'tags': 'tags.csv',
                'links': 'links.csv',
                'genome_scores': 'genome-scores.csv',
                'genome_tags': 'genome-tags.csv'
            }
        },
        'ml-25m': {
            'url': 'ml-25m.zip',
            'extract_dir': 'ml-25m',
            'size': '250MB',
            'ratings': 25000000,
            'has_tags': True,
            'file_mappings': {
                'ratings': 'ratings.csv',
                'movies': 'movies.csv',
                'tags': 'tags.csv',
                'links': 'links.csv',
                'genome_scores': 'genome-scores.csv',
                'genome_tags': 'genome-tags.csv'
            }
        }
    }

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        log_level: int = 0
    ) -> None:
        """Initialize downloader with cache directory and logging level."""
        # Setup logging
        self.logger = get_logger(log_level, self)

        # Setup cache directory
        cache_path = Path(cache_dir or tempfile.gettempdir())
        self.cache_dir = cache_path / "movielens"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"Cache directory: {self.cache_dir}")

    def download_dataset(
        self,
        dataset: str = 'ml-20m',
        force_download: bool = False
    ) -> Path:
        """Download and extract MovieLens dataset, returns path to dataset."""
        self.logger.debug(f"Downloading dataset: {dataset}")

        if dataset not in self.DATASETS:
            available = list(self.DATASETS.keys())
            error_msg = f"Dataset {dataset} not supported. Available: {available}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Setup paths
        dataset_info = self.DATASETS[dataset]
        zip_path = self.cache_dir / f"{dataset}.zip"
        extract_path = self.cache_dir / dataset_info['extract_dir']

        # Check if already downloaded and extracted
        if extract_path.exists() and not force_download:
            self.logger.info(
                f"Dataset {dataset} already exists at {extract_path}")
            return extract_path

        # Download dataset
        url = f"{self.BASE_URL}{dataset_info['url']}"
        size = dataset_info['size']
        self.logger.info(f"Downloading {dataset} ({size}) from {url}...")

        # Stream download with progress
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()
        self.logger.debug(f"Download response status: {response.status_code}")

        # Save zip file
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        self.logger.debug(f"Saved zip to: {zip_path}")

        # Extract zip file
        self.logger.info(f"Extracting {dataset}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.cache_dir)

        self.logger.debug(f"Extracted to: {extract_path}")
        return extract_path

    def _load_file(
        self,
        file_path: Path,
        dataset: str
    ) -> Optional[pd.DataFrame]:
        """Load file based on extension and dataset type, returns DataFrame or None."""
        try:
            if file_path.suffix == '.csv':
                return pd.read_csv(file_path)
            elif file_path.suffix == '.dat':
                # Handle different .dat formats per dataset
                if dataset == 'ml-1m':
                    if 'ratings' in file_path.name:
                        return pd.read_csv(
                            file_path,
                            sep='::',
                            names=['userId', 'movieId', 'rating', 'timestamp'],
                            engine='python'
                        )
                    elif 'movies' in file_path.name:
                        return pd.read_csv(
                            file_path,
                            sep='::',
                            names=['movieId', 'title', 'genres'],
                            engine='python',
                            encoding='latin1'
                        )
                    elif 'users' in file_path.name:
                        return pd.read_csv(
                            file_path,
                            sep='::',
                            names=['userId', 'gender', 'age',
                                   'occupation', 'zipCode'],
                            engine='python'
                        )
                elif dataset == 'ml-10m':
                    if 'ratings' in file_path.name:
                        return pd.read_csv(
                            file_path,
                            sep='::',
                            names=['userId', 'movieId', 'rating', 'timestamp'],
                            engine='python'
                        )
                    elif 'movies' in file_path.name:
                        return pd.read_csv(
                            file_path,
                            sep='::',
                            names=['movieId', 'title', 'genres'],
                            engine='python',
                            encoding='latin1'
                        )
                    elif 'tags' in file_path.name:
                        return pd.read_csv(
                            file_path,
                            sep='::',
                            names=['userId', 'movieId', 'tag', 'timestamp'],
                            engine='python'
                        )
            elif dataset == 'ml-100k' and file_path.suffix == '.data':
                return pd.read_csv(
                    file_path,
                    sep='\t',
                    names=['userId', 'movieId', 'rating', 'timestamp']
                )
            elif dataset == 'ml-100k' and file_path.suffix == '.item':
                return pd.read_csv(
                    file_path,
                    sep='|',
                    names=['movieId', 'title', 'release_date', 'video_release_date', 'imdb_url'] +
                          [f'genre_{i}' for i in range(19)],
                    encoding='latin1'
                )
            elif dataset == 'ml-100k' and file_path.suffix == '.user':
                return pd.read_csv(
                    file_path,
                    sep='|',
                    names=['userId', 'age', 'gender', 'occupation', 'zipCode']
                )
            else:
                self.logger.warning(f"Unknown file format: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None

    def load_dataset_with_tags(
        self,
        dataset: str = 'ml-20m'
    ) -> Dict[str, pd.DataFrame]:
        """Load complete MovieLens dataset with all files, returns dict of DataFrames."""
        self.logger.debug(f"Loading dataset with tags: {dataset}")

        # Download dataset if needed
        dataset_path = self.download_dataset(dataset)
        data = {}

        # Get dataset-specific file mappings
        dataset_info = self.DATASETS[dataset]
        file_mappings = dataset_info['file_mappings']

        # Use the actual extracted directory path
        actual_data_path = dataset_path
        self.logger.debug(f"Data path: {actual_data_path}")
        if actual_data_path.exists():
            contents = list(actual_data_path.iterdir())
            self.logger.debug(
                f"Data directory contents: {[p.name for p in contents]}")

        # Load each file if it exists
        for key, filename in file_mappings.items():
            file_path = actual_data_path / filename
            if file_path.exists():
                df = self._load_file(file_path, dataset)
                if df is not None:
                    data[key] = df
                    self.logger.info(f"Loaded {len(data[key]):,} {key}")
                else:
                    self.logger.warning(f"Failed to load {filename}")
            else:
                self.logger.debug(f"File not found: {file_path}")

        self.logger.info(f"Loaded {len(data)} datasets: {list(data.keys())}")
        return data

    def get_user_tags(self, tags_df: pd.DataFrame, user_id: int) -> pd.DataFrame:
        """Get all tags applied by a specific user."""
        self.logger.debug(f"Getting tags for user {user_id}")
        result = tags_df[tags_df['userId'] == user_id]
        self.logger.debug(f"Found {len(result)} tags for user {user_id}")
        return result

    def get_movie_tags(self, tags_df: pd.DataFrame, movie_id: int) -> pd.DataFrame:
        """Get all tags applied to a specific movie."""
        self.logger.debug(f"Getting tags for movie {movie_id}")
        result = tags_df[tags_df['movieId'] == movie_id]
        self.logger.debug(f"Found {len(result)} tags for movie {movie_id}")
        return result

    def print_dataset_summary(self, data: Dict[str, pd.DataFrame]) -> None:
        """Print summary statistics for all loaded DataFrames."""
        self.logger.info("Dataset summary:")

        # Define columns to check and their display names
        columns_to_check = {
            'userId': 'Unique Users',
            'movieId': 'Unique Movies',
            'tag': 'Unique Tags',
            'tagId': 'Unique Tag IDs',
            'genres': 'Unique Genres',
            'title': 'Unique Titles'
        }

        for df_name, df in data.items():
            print(f"{df_name.upper()}" + "-" * 20)
            print(f"Total rows: {len(df):,}")

            # Check each potential column
            found_columns = []
            for col, display_name in columns_to_check.items():
                if col in df.columns:
                    unique_count = df[col].nunique()
                    print(f"{display_name}: {unique_count:,}")
                    found_columns.append(col)

            # Show sample columns if no standard ones found
            if not found_columns:
                print(f"Columns: {list(df.columns)}")
                if len(df.columns) <= 5:
                    for col in df.columns[:3]:  # Show first 3 columns
                        if df[col].dtype == 'object' or df[col].dtype.name.startswith('int'):
                            unique_count = df[col].nunique()
                            print(f"Unique {col}: {unique_count:,}")


def main() -> None:
    """Download MovieLens 20M dataset and display summary statistics."""
    # Setup logging for main
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize downloader with proper log level (int, not string)
    downloader = MovieLensDownloader(log_level=logging.INFO)
    logger.info("Initialized MovieLens downloader")

    # Load MovieLens 20M with tags
    logger.info("Loading MovieLens 20M dataset with tags...")
    data = downloader.load_dataset_with_tags('ml-20m')

    # Display dataset summary (this actually shows useful information)
    downloader.print_dataset_summary(data)


if __name__ == "__main__":
    main()
