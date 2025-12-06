"""
Intelligent Molecular Dataset Splitting Tool (Fixed Version)
Splits dataset into training, validation, and test sets (8:1:1) based on molecular structure differences
Resolves data leakage issues
"""

import os
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MolecularDatasetSplitter:
    """Dataset splitter based on molecular structure differences"""

    def __init__(self, csv_path, smiles_column='smiles', seed=42, handle_duplicates='remove'):
        """
        Initialize splitter

        Args:
            csv_path: CSV file path
            smiles_column: SMILES column name
            seed: Random seed
            handle_duplicates: Method to handle duplicates ('remove', 'keep_first', 'average')
        """
        self.seed = seed
        np.random.seed(seed)

        # Read data
        self.df = pd.read_csv(csv_path)
        self.smiles_column = smiles_column
        self.handle_duplicates = handle_duplicates

        # Validate SMILES
        self._validate_smiles()

        # Handle duplicate SMILES
        self._handle_duplicates()

        print(f"Successfully loaded {len(self.df)} molecules")

    def _validate_smiles(self):
        """Validate and clean invalid SMILES"""
        valid_indices = []
        invalid_smiles = []

        for idx, smiles in enumerate(self.df[self.smiles_column]):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                valid_indices.append(idx)
            else:
                invalid_smiles.append(smiles)

        initial_size = len(self.df)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        removed = initial_size - len(self.df)

        if removed > 0:
            print(f"Removed {removed} invalid molecules")
            if len(invalid_smiles) <= 5:
                print(f"Invalid SMILES: {invalid_smiles}")

    def _handle_duplicates(self):
        """Handle duplicate SMILES"""
        initial_size = len(self.df)

        # Check duplicates
        duplicates = self.df[self.smiles_column].duplicated()
        n_duplicates = duplicates.sum()

        if n_duplicates > 0:
            print(f"Found {n_duplicates} duplicate SMILES")

            if self.handle_duplicates == 'remove':
                # Keep first occurrence, remove duplicates
                self.df = self.df[~duplicates].reset_index(drop=True)
                print(f"After removing duplicates, {len(self.df)} unique molecules remain")

            elif self.handle_duplicates == 'keep_first':
                # Keep first occurrence (including its property values)
                self.df = self.df.drop_duplicates(subset=[self.smiles_column], keep='first').reset_index(drop=True)
                print(f"Kept first occurrence of molecules, {len(self.df)} remaining")

            elif self.handle_duplicates == 'average':
                # For numerical columns, calculate average
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    # Group and calculate average
                    agg_dict = {col: 'mean' for col in numeric_cols}
                    # Keep first value of non-numeric columns
                    non_numeric_cols = [col for col in self.df.columns
                                       if col not in numeric_cols and col != self.smiles_column]
                    for col in non_numeric_cols:
                        agg_dict[col] = 'first'

                    self.df = self.df.groupby(self.smiles_column).agg(agg_dict).reset_index()
                    print(f"Averaged numerical properties for duplicate molecules, {len(self.df)} unique molecules remain")
                else:
                    # If no numerical columns, keep first
                    self.df = self.df.drop_duplicates(subset=[self.smiles_column], keep='first').reset_index(drop=True)
                    print(f"Kept first occurrence of molecules, {len(self.df)} remaining")
        else:
            print("No duplicate SMILES found")

    def compute_morgan_fingerprints(self, radius=2, n_bits=2048):
        """Compute Morgan fingerprints"""
        print("Computing Morgan fingerprints...")
        fingerprints = []
        for smiles in tqdm(self.df[self.smiles_column]):
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(np.array(fp))
        return np.array(fingerprints)

    def compute_scaffolds(self):
        """Extract molecular scaffolds"""
        print("Extracting molecular scaffolds...")
        scaffolds = []
        for smiles in tqdm(self.df[self.smiles_column]):
            mol = Chem.MolFromSmiles(smiles)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
            scaffolds.append(scaffold)
        return scaffolds

    def diversity_based_split(self, train_ratio=0.8, val_ratio=0.1, method='maxmin'):
        """
        Split dataset based on molecular diversity

        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            method: Splitting method ('maxmin', 'scaffold', 'clustering', 'random')

        Returns:
            train_df, val_df, test_df
        """
        test_ratio = 1 - train_ratio - val_ratio
        n_total = len(self.df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        if method == 'maxmin':
            return self._maxmin_split(n_train, n_val, n_test)
        elif method == 'scaffold':
            return self._scaffold_split(train_ratio, val_ratio)
        elif method == 'clustering':
            return self._clustering_split(n_train, n_val, n_test)
        elif method == 'random':
            return self._random_split(n_train, n_val, n_test)
        else:
            raise ValueError(f"Unknown splitting method: {method}")

    def _random_split(self, n_train, n_val, n_test):
        """Random split (as baseline)"""
        print("Using random split...")

        # Randomly shuffle indices
        indices = np.arange(len(self.df))
        np.random.shuffle(indices)

        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]

        # Create dataframes
        train_df = self.df.iloc[train_indices].reset_index(drop=True)
        val_df = self.df.iloc[val_indices].reset_index(drop=True)
        test_df = self.df.iloc[test_indices].reset_index(drop=True)

        # Verify no overlap
        self._verify_no_overlap(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def _maxmin_split(self, n_train, n_val, n_test):
        """
        MaxMin algorithm: Select molecules with maximum mutual differences as test set
        """
        print("Using MaxMin algorithm for splitting...")

        # Compute fingerprints
        fps = self.compute_morgan_fingerprints()

        # Compute distance matrix (using Tanimoto similarity)
        print("Computing inter-molecular distances...")
        distances = cdist(fps, fps, metric='jaccard')

        # MaxMin sampling to select test set
        selected_indices = []
        remaining_indices = list(range(len(self.df)))

        # Randomly select first molecule
        first_idx = np.random.choice(remaining_indices)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Select molecules with maximum difference from already selected molecules
        print("Selecting molecules with maximum differences as test set...")
        for _ in tqdm(range(n_test - 1)):
            if not remaining_indices:
                break

            # Compute minimum distance from remaining molecules to selected molecules
            min_distances = []
            for idx in remaining_indices:
                min_dist = min(distances[idx, selected_idx]
                              for selected_idx in selected_indices)
                min_distances.append(min_dist)

            # Select molecule with maximum minimum distance (most dissimilar)
            max_min_idx = np.argmax(min_distances)
            selected_idx = remaining_indices[max_min_idx]

            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)

        test_indices = selected_indices

        # Select validation set (molecules with medium differences)
        val_indices = []
        for _ in range(n_val):
            if not remaining_indices:
                break

            # Compute average distance to test set
            avg_distances = []
            for idx in remaining_indices:
                avg_dist = np.mean([distances[idx, test_idx]
                                   for test_idx in test_indices])
                avg_distances.append(avg_dist)

            # Select molecules with medium distance
            median_idx = np.argsort(avg_distances)[len(avg_distances)//2]
            selected_idx = remaining_indices[median_idx]

            val_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)

        # Remaining as training set
        train_indices = remaining_indices

        # Create split dataframes
        train_df = self.df.iloc[train_indices].reset_index(drop=True)
        val_df = self.df.iloc[val_indices].reset_index(drop=True)
        test_df = self.df.iloc[test_indices].reset_index(drop=True)

        # Verify no overlap
        self._verify_no_overlap(train_df, val_df, test_df)

        # Calculate and report diversity metrics
        self._report_split_statistics(train_indices, val_indices, test_indices, distances)

        return train_df, val_df, test_df

    def _scaffold_split(self, train_ratio, val_ratio):
        """
        Scaffold-based splitting
        """
        print("Using scaffold split...")

        # Extract scaffolds
        scaffolds = self.compute_scaffolds()

        # Group by scaffold
        scaffold_dict = {}
        for idx, scaffold in enumerate(scaffolds):
            if scaffold not in scaffold_dict:
                scaffold_dict[scaffold] = []
            scaffold_dict[scaffold].append(idx)

        # Sort by scaffold size
        scaffold_list = sorted(scaffold_dict.items(),
                              key=lambda x: len(x[1]),
                              reverse=True)

        # Assign scaffolds to different sets
        train_indices, val_indices, test_indices = [], [], []
        train_size = int(len(self.df) * train_ratio)
        val_size = int(len(self.df) * val_ratio)

        for scaffold, indices in scaffold_list:
            if len(train_indices) < train_size:
                train_indices.extend(indices)
            elif len(val_indices) < val_size:
                val_indices.extend(indices)
            else:
                test_indices.extend(indices)

        train_df = self.df.iloc[train_indices].reset_index(drop=True)
        val_df = self.df.iloc[val_indices].reset_index(drop=True)
        test_df = self.df.iloc[test_indices].reset_index(drop=True)

        # Verify no overlap
        self._verify_no_overlap(train_df, val_df, test_df)

        print(f"Scaffold statistics: Total {len(scaffold_dict)} different scaffolds")
        print(f"Training set scaffolds: {len(set(scaffolds[i] for i in train_indices))} scaffolds")
        print(f"Validation set scaffolds: {len(set(scaffolds[i] for i in val_indices))} scaffolds")
        print(f"Test set scaffolds: {len(set(scaffolds[i] for i in test_indices))} scaffolds")

        return train_df, val_df, test_df

    def _clustering_split(self, n_train, n_val, n_test):
        """
        Clustering-based splitting
        """
        print("Using clustering split...")

        # Compute fingerprints
        fps = self.compute_morgan_fingerprints()

        # Clustering
        n_clusters = min(10, len(self.df) // 10)  # Adaptive number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        cluster_labels = kmeans.fit_predict(fps)

        # Compute cluster centers
        cluster_centers = []
        cluster_indices = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_idx = np.where(cluster_mask)[0]
            cluster_indices[i] = cluster_idx.tolist()

            if len(cluster_idx) > 0:
                center = fps[cluster_mask].mean(axis=0)
                cluster_centers.append(center)
            else:
                cluster_centers.append(np.zeros(fps.shape[1]))

        cluster_centers = np.array(cluster_centers)

        # Compute inter-cluster distances
        cluster_distances = cdist(cluster_centers, cluster_centers, metric='jaccard')

        # Select clusters with maximum mutual distances as test set
        test_clusters = []
        test_indices = []

        while len(test_indices) < n_test and len(test_clusters) < n_clusters:
            if not test_clusters:
                # Start with smallest cluster
                cluster_id = min(cluster_indices.keys(), key=lambda x: len(cluster_indices[x]))
            else:
                # Select cluster with maximum distance from selected clusters
                remaining_clusters = [i for i in range(n_clusters) if i not in test_clusters]
                if not remaining_clusters:
                    break

                max_min_dist = -1
                cluster_id = -1

                for c in remaining_clusters:
                    if len(cluster_indices[c]) == 0:
                        continue
                    min_dist = min(cluster_distances[c, t] for t in test_clusters)
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        cluster_id = c

                if cluster_id == -1:
                    break

            test_clusters.append(cluster_id)
            test_indices.extend(cluster_indices[cluster_id])

        test_indices = test_indices[:n_test]

        # Select validation and training sets
        remaining_indices = [i for i in range(len(self.df)) if i not in test_indices]
        np.random.shuffle(remaining_indices)

        val_indices = remaining_indices[:n_val]
        train_indices = remaining_indices[n_val:]

        train_df = self.df.iloc[train_indices].reset_index(drop=True)
        val_df = self.df.iloc[val_indices].reset_index(drop=True)
        test_df = self.df.iloc[test_indices].reset_index(drop=True)

        # Verify no overlap
        self._verify_no_overlap(train_df, val_df, test_df)

        print(f"Clustering statistics: {n_clusters} clusters")
        print(f"Test set from {len(test_clusters)} clusters")

        return train_df, val_df, test_df

    def _verify_no_overlap(self, train_df, val_df, test_df):
        """Verify no duplicate SMILES between datasets"""
        train_smiles = set(train_df[self.smiles_column])
        val_smiles = set(val_df[self.smiles_column])
        test_smiles = set(test_df[self.smiles_column])

        # Check intersections
        train_val = train_smiles & val_smiles
        train_test = train_smiles & test_smiles
        val_test = val_smiles & test_smiles

        if train_val:
            print(f"Warning: Training and validation sets have {len(train_val)} duplicate molecules")
            print(f"Duplicate SMILES examples: {list(train_val)[:3]}")

        if train_test:
            print(f"Warning: Training and test sets have {len(train_test)} duplicate molecules")
            print(f"Duplicate SMILES examples: {list(train_test)[:3]}")

        if val_test:
            print(f"Warning: Validation and test sets have {len(val_test)} duplicate molecules")
            print(f"Duplicate SMILES examples: {list(val_test)[:3]}")

        if not (train_val or train_test or val_test):
            print("✓ Dataset split successful: No duplicate molecules")

        return not (train_val or train_test or val_test)

    def _report_split_statistics(self, train_idx, val_idx, test_idx, distances):
        """Report splitting statistics"""
        print("\n=== Split Statistics ===")
        print(f"Training set: {len(train_idx)} molecules")
        print(f"Validation set: {len(val_idx)} molecules")
        print(f"Test set: {len(test_idx)} molecules")

        # Calculate average distances between sets
        if len(test_idx) > 0 and len(train_idx) > 0:
            train_test_dist = distances[np.ix_(train_idx, test_idx)].mean()
            print(f"\nTrain-test average distance: {train_test_dist:.3f}")

        if len(val_idx) > 0 and len(train_idx) > 0:
            train_val_dist = distances[np.ix_(train_idx, val_idx)].mean()
            print(f"Train-validation average distance: {train_val_dist:.3f}")

        if len(val_idx) > 0 and len(test_idx) > 0:
            val_test_dist = distances[np.ix_(val_idx, test_idx)].mean()
            print(f"Validation-test average distance: {val_test_dist:.3f}")

        # Calculate internal average distances
        if len(train_idx) > 1:
            train_internal = distances[np.ix_(train_idx, train_idx)][np.triu_indices(len(train_idx), 1)].mean()
            print(f"\nTraining set internal average distance: {train_internal:.3f}")

        if len(test_idx) > 1:
            test_internal = distances[np.ix_(test_idx, test_idx)][np.triu_indices(len(test_idx), 1)].mean()
            print(f"Test set internal average distance: {test_internal:.3f}")

def analyze_split_similarity(train_path, val_path, test_path, output_dir):
    """Analyze similarity of split datasets (improved version)"""

    # First verify no duplicates
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Assume first column is SMILES
    smiles_col = train_df.columns[0]

    train_smiles = set(train_df[smiles_col])
    val_smiles = set(val_df[smiles_col])
    test_smiles = set(test_df[smiles_col])

    # Check duplicates
    overlaps = {
        'train-val': train_smiles & val_smiles,
        'train-test': train_smiles & test_smiles,
        'val-test': val_smiles & test_smiles
    }

    print("\n=== Dataset Overlap Check ===")
    has_overlap = False
    for key, overlap in overlaps.items():
        if overlap:
            has_overlap = True
            print(f"{key}: {len(overlap)} duplicate molecules")
        else:
            print(f"{key}: ✓ No duplicates")

    if not has_overlap:
        print("\n✓ No duplicate molecules between all datasets")

        # If find_similar_mols.py exists, perform similarity analysis
        if os.path.exists('find_similar_mols.py'):
            import subprocess

            similarity_report = os.path.join(output_dir, 'test_train_similarity.csv')

            cmd = [
                'python', 'find_similar_mols.py',
                '--test_path', test_path,
                '--train_path', train_path,
                '--save_path', similarity_report,
                '--distance_measure', 'morgan',
                '--num_neighbors', '5'
            ]

            try:
                subprocess.run(cmd, check=True)

                # Read and analyze results
                sim_df = pd.read_csv(similarity_report)

                # Calculate statistics
                avg_distance = sim_df['train_1_morgan_jaccard_distance'].mean()
                print(f"\nTest set nearest neighbor average distance to training set: {avg_distance:.3f}")

                # High-difference molecules (distance > 0.7)
                high_diff = sim_df[sim_df['train_1_morgan_jaccard_distance'] > 0.7]
                print(f"Number of high-difference molecules: {len(high_diff)} ({len(high_diff)/len(sim_df)*100:.1f}%)")

            except Exception as e:
                print(f"Similarity analysis failed: {e}")
    else:
        print("\n⚠️ Warning: Detected duplicates between datasets, please check data processing pipeline")

def main():
    parser = argparse.ArgumentParser(description='Molecular dataset splitting tool based on diversity')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--output_dir', type=str, default='./splits',
                        help='Output directory')
    parser.add_argument('--smiles_column', type=str, default='smiles',
                        help='SMILES column name')
    parser.add_argument('--method', type=str, default='maxmin',
                        choices=['maxmin', 'scaffold', 'clustering', 'random'],
                        help='Splitting method')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--prefix', type=str, default='split',
                        help='Output file prefix')
    parser.add_argument('--handle_duplicates', type=str, default='remove',
                        choices=['remove', 'keep_first', 'average'],
                        help='Method to handle duplicate SMILES')

    args = parser.parse_args()

    # Create splitter
    splitter = MolecularDatasetSplitter(
        csv_path=args.input,
        smiles_column=args.smiles_column,
        seed=args.seed,
        handle_duplicates=args.handle_duplicates
    )

    # Execute splitting
    train_df, val_df, test_df = splitter.diversity_based_split(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        method=args.method
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, f'{args.prefix}_train.csv')
    val_path = os.path.join(args.output_dir, f'{args.prefix}_val.csv')
    test_path = os.path.join(args.output_dir, f'{args.prefix}_test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nResults saved to {args.output_dir}")
    print(f"  Training set: {train_path} ({len(train_df)} molecules)")
    print(f"  Validation set: {val_path} ({len(val_df)} molecules)")
    print(f"  Test set: {test_path} ({len(test_df)} molecules)")

    # Generate similarity analysis report
    print("\nGenerating split quality report...")
    analyze_split_similarity(train_path, val_path, test_path, args.output_dir)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage example:")
        print("python molecular_split.py --input data.csv --output_dir ./splits --method maxmin")
        print("\nHandling duplicate molecules:")
        print("python molecular_split.py --input data.csv --handle_duplicates remove")