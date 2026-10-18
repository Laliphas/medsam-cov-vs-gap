"""Task 1: reproduce MedSAM + covariance + SPDNet on DermaMNIST (official split, debug off).

    python src/reproduce_derma.py

Runs the official main.handle_spdnet_gve for MedSAM only (the `gves` argument is our patch),
then copies the 5 runs into results/derma_spdnet/ and writes the evaluation report.
"""
import os

import repo

from train import SHORT_NAMES, export_runs
import eval_report


def main():
    import main as official                  # official pipeline; loads config.json and sets the device
    from experiments import datasets, to_dataframe

    derma = next(d for d in datasets if d.dataset_name == 'dermamnist')
    official.handle_spdnet_gve(derma, gves=('medsam',), debug=False)
    for (technique, features), results in official.reports.items():
        print(technique, features)
        print(to_dataframe(results).to_string())

    run_dir = os.path.join(official.user_config.model_dir, 'SPDNet', 'dermamnist', 'medsam')
    out_dir = os.path.join(repo.RESULTS, 'derma_spdnet')
    export_runs(run_dir, out_dir)
    eval_report.report(os.path.join(out_dir, 'runs'), out_dir, SHORT_NAMES['derma'],
                       'DermaMNIST - MedSAM + covariance + SPDNet (official code)')


if __name__ == '__main__':
    main()
