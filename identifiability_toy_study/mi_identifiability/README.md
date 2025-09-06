# MI-identifiability

Experiments from the paper can be reproduced with the following commands (some variability is to be expected due to various hardware configurations):

## Main results

### Basic setup (also see notebook)
```bash
main.py --verbose --val-frequency 1 --noise-std 0.0 --target-logic-gates XOR --n-experiments 100
```

### Multi-task training
```bash
main.py --verbose --val-frequency 1 --noise-std 0.0 --n-gates 1 2 3 4 5 6 --n-experiments 100
```

### Varying the target gate
```bash
main.py --verbose --val-frequency 1 --noise-std 0.0 --target-logic-gates XOR AND OR IMP --n-experiments 100
```

### Varying the network's size
```bash
main.py --verbose --val-frequency 1 --noise-std 0.0 --target-logic-gates XOR --n-experiments 100 --size 2 3 4 5
```

## Additional results

### Effect of noisy training
These results are to be compared with those of the basic setup.
```bash
main.py --verbose --val-frequency 1 --noise-std 0.1 --target-logic-gates XOR --n-experiments 100
```

### Effect of skewed distributions
These results are to be compared with those of the basic setup.
```bash
main.py --verbose --val-frequency 1 --noise-std 0.0 --target-logic-gates XOR --n-experiments 100 --skewed-distribution
```

### Effect of target loss
```bash
main.py --verbose --val-frequency 1 --noise-std 0.0 --target-logic-gates XOR --n-experiments 100 --loss-target 0.1 0.01 0.001 0.0001 0.00001 0.000001
```
