import os
import pandas as pd

def make_leaderboard(scores, output_path=None, args=None):
    """Make a leaderboard out of individual synthesizer scores.
    Args:
        scores (list):
            List of DataFrames with scores or paths to CSV files.
        output_path (str):
            If an ``output_path`` is given, the generated leaderboard will be stored in the
            indicated path as a CSV file. The given path must be a complete path including
            the ``.csv`` filename.
    Returns:
        pandas.DataFrame or None:
            If not ``output_path`` is given, a table containing one row per synthesizer and
            one column for each dataset and metric is returned. Otherwise, there is no output.
    """
    if isinstance(scores, str) and os.path.isdir(scores):
        scores = [
            pd.read_csv(os.path.join(scores, path), index_col=0)
            for path in os.listdir(scores)
        ]
    try:
        scores = pd.concat(scores, ignore_index=True)
        scores = scores.drop(['distance', 'iteration', 'name'], axis=1, errors='ignore')
    except:
        return scores
    leaderboard = scores.mean()

    if output_path:
        os.makedirs(os.path.dirname(os.path.realpath(output_path)), exist_ok=True)
        leaderboard.to_csv(output_path)

    return leaderboard
    