import argparse
import datetime
import os
import random
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mi_identifiability.circuit import find_circuits
from mi_identifiability.mappings import find_minimal_mappings
from mi_identifiability.logic_gates import ALL_LOGIC_GATES, generate_noisy_multi_gate_data, get_formula_dataset
from mi_identifiability.neural_model import MLP
from mi_identifiability.utils import setup_logging, set_seeds


def run_experiment(logger, output_dir, run_dir, args):
    data = []

    if args.resume_from is not None:
        # Read output from previous run
        prev_run_dir = output_dir / args.resume_from
        in_file = prev_run_dir / 'data_tmp.csv'
        try:
            run_dir = prev_run_dir
            data = pd.read_csv(in_file).to_dict('records')
            in_file.rename(prev_run_dir / 'data_tmp.bak')
        except FileNotFoundError:
            logger.error(f'Previous run data {args.resume_from} not found, starting from scratch')

    all_logic_gates = [ALL_LOGIC_GATES[gate] for gate in args.target_logic_gates]
    formula_dataset = {depth: get_formula_dataset(all_logic_gates, max_depth=depth + 1) for depth in args.depth}

    for n_gates, seed_offset in product(args.n_gates, range(args.n_experiments)):
        gates = random.sample(all_logic_gates, k=n_gates)
        n_inputs = gates[0].n_inputs

        if n_gates > 1:
            x_tt, y_tt = generate_noisy_multi_gate_data(gates, n_repeats=args.n_repeats, noise_std=args.noise_std,
                                                        device=args.device)
        else:
            x_tt, y_tt = gates[0].generate_noisy_data(n_repeats=args.n_repeats, noise_std=args.noise_std,
                                                      device=args.device)

        print(f'Iteration # {seed_offset}')
        weights = None
        if args.skewed_distribution:
            weights = [random.random() for _ in range(2 ** n_inputs)]

        if n_gates > 1:
            x, y = generate_noisy_multi_gate_data(gates, n_repeats=args.n_samples_train, weights=weights,
                                                  noise_std=args.noise_std, device=args.device)
            x_val, y_val = generate_noisy_multi_gate_data(gates, n_repeats=args.n_samples_val, weights=weights,
                                                          noise_std=args.noise_std, device=args.device)
        else:
            x, y = gates[0].generate_noisy_data(n_repeats=args.n_samples_train, weights=weights,
                                                noise_std=args.noise_std, device=args.device)
            x_val, y_val = gates[0].generate_noisy_data(n_repeats=args.n_samples_val, weights=weights,
                                                        noise_std=args.noise_std, device=args.device)

        for k, depth, loss_target, lr in product(args.size, args.depth, args.loss_target, args.learning_rate):
            set_seeds(args.seed + seed_offset)

            skip_exp = False

            for e in data:
                if (e['gates'] == ' '.join(lg.name for lg in gates) and e['n_gates'] >= n_gates and e['sizes'] >= k and
                        e['seed_offset'] >= seed_offset and e['loss_target'] >= loss_target and e['learning_rate'] >= lr
                        and e['depth'] >= depth):
                    print('Skipping already recorded run')
                    skip_exp = True
                    break

            if skip_exp:
                continue

            layer_sizes = [2] + [k] * depth + [n_gates]
            model = MLP(hidden_sizes=layer_sizes[1:-1], input_size=n_inputs, output_size=n_gates, device=args.device)

            avg_loss = model.do_train(
                x=x,
                y=y,
                x_val=x_val,
                y_val=y_val,
                batch_size=args.batch_size,
                learning_rate=lr,
                epochs=args.epochs,
                loss_target=loss_target,
                val_frequency=args.val_frequency,
                logger=logger if args.verbose else None
            )

            val_acc = model.do_eval(x_val, y_val)

            if val_acc < 0.99 or avg_loss > loss_target:
                logger.info(f"No convergence - k={k}, seed={seed_offset}, loss={avg_loss}/{loss_target}, acc={val_acc}")
                continue

            min_sparsity = args.min_sparsity
            if k > 3:
                min_sparsity = 0.2
            if k > 4:
                min_sparsity = 0.3

            n_circuits = []
            total_n_groundings = []
            n_formulas = []
            total_n_mappings = []

            submodels = model.separate_into_k_mlps()
            for i, submodel in enumerate(submodels):
                top_circuits, sparsities, _ = find_circuits(
                    submodel,
                    x_val,
                    y_val[..., i].unsqueeze(1),
                    accuracy_threshold=args.accuracy_threshold,
                    min_sparsity=min_sparsity
                )

                n_circuits.append(len(top_circuits))

                if args.max_circuits is not None and args.max_circuits < len(top_circuits):
                    sample_top_circuits = [c for c, _ in sorted(zip(top_circuits, sparsities),
                                                                key=lambda t: t[1], reverse=True)[:args.max_circuits]]
                else:
                    sample_top_circuits = top_circuits

                n_groundings = []
                logger.info("Grounding circuits")
                for circuit in tqdm(sample_top_circuits):
                    gds = circuit.ground(submodel(x_tt, circuit=circuit, return_activations=True))
                    n_groundings.append(len(gds))

                total_n_groundings.append(n_groundings)

                print(f'Circuits: {n_circuits}, groundings: {total_n_groundings}')

                formulas = set()
                n_mappings = []
                for idx, formula_data in enumerate(formula_dataset[depth][gates[i].name]):
                    minimal_mappings = find_minimal_mappings(submodel, *formula_data)

                    if minimal_mappings:
                        formulas.add(formula_data[0])
                        n_mappings.append(len(minimal_mappings))
                n_formulas.append(len(formulas))
                total_n_mappings.append(n_mappings)

                print(f'Formulas: {n_formulas}, mappings: {total_n_mappings}')

            data.append({
                'gates': ' '.join(lg.name for lg in gates),
                'n_gates': n_gates,
                'sizes': k,
                'seed_offset': seed_offset,
                'perfect_circuits': n_circuits,
                'interpretations_per_circuit': total_n_groundings,
                'formulas': n_formulas,
                'mappings_per_formula': total_n_mappings,
                'loss_target': loss_target,
                'learning_rate': lr,
                'depth': depth,
                'weights': weights,
            })
            pd.DataFrame(data).to_csv(os.path.join(run_dir, 'data_tmp.csv'), index=False)

    return run_dir, pd.DataFrame(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='Starting random seed')
    parser.add_argument('--size', type=int, nargs='+', default=[3],
                        help='Size of the hidden layers of the neural networks')
    parser.add_argument('--depth', type=int, nargs='+', default=[2],
                        help='Number of hidden layers in the neural networks')
    parser.add_argument('--n-repeats', type=int, default=10,
                        help='Number of times to repeat the data for circuit search')
    parser.add_argument('--n-experiments', type=int, default=100,
                        help='Runs each experiment n times with different seeds')
    parser.add_argument('--noise-std', type=float, default=0.1, help='Standard deviation of the Gaussian noise')
    parser.add_argument('--n-samples-train', type=int, default=1000, help='Number of samples for training')
    parser.add_argument('--n-samples-val', type=int, default=50, help='Number of samples for validation')
    parser.add_argument('--n-gates', type=int, nargs='+', default=[1],
                        help='Number of logic gates used as target functions')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, nargs='+', default=[0.001],
                        help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--max-circuits', type=int, default=None, help='Maximum number of circuits used for grounding')
    parser.add_argument('--min-sparsity', type=float, default=0.0, help='Minimum sparsity for circuit search')
    parser.add_argument('--loss-target', type=float, nargs='+', default=[0.01],
                        help='Target loss value for training')
    parser.add_argument('--skewed-distribution', action='store_true',
                        help='Whether to use a skewed distribution for training data')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to store the data on')
    parser.add_argument('--target-logic-gates', type=str, nargs='+',
                        default=['AND', 'OR', 'IMP', 'NAND', 'NOR', 'NIMP', 'XOR'], choices=ALL_LOGIC_GATES.keys(),
                        help='The allowed target logic gates')
    parser.add_argument('--accuracy-threshold', type=float, default=0.99,
                        help='The accuracy threshold for circuit search')
    parser.add_argument('--val-frequency', type=int, default=10, help='Frequency of validation during training')
    parser.add_argument('--verbose', action='store_true', help='Whether to display extra information')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from a previous run')

    args = parser.parse_args()
    output_dir = Path('logs')

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"

    if args.resume_from is None:
        logger = setup_logging(run_dir, "output.log")
    else:
        logger = setup_logging(output_dir / args.resume_from, "output.log")

    logger.info("Configuration in use:")
    logger.info(args)

    seed = args.seed
    set_seeds(seed)
    logger.info(f"Setting the seeds: {seed}")

    run_dir, df_res = run_experiment(logger, output_dir, run_dir, args)

    output_filepath = run_dir / "df_out.csv"
    df_res.to_csv(output_filepath)
    logger.info(f"Results were saved at {output_filepath}")

    print(df_res)
