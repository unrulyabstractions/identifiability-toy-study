# MI-Identifiability Research Analysis Notes

## Overview

This research investigates the **identifiability problem in Mechanistic Interpretability (MI)**, specifically examining whether different MI approaches yield consistent explanations when applied to the same neural network. The core finding appears to be that MI criteria are **not uniquely identifiable** - different valid explanations can conflict with each other.

## Core Research Question

**Can we uniquely identify the "true" mechanistic explanation of a neural network's behavior?**

The answer appears to be: **No** - multiple conflicting but equally valid explanations can coexist.

## Two Main MI Strategies Studied

### 1. What-Then-Where Strategy
- **Approach**: Start with high-level algorithms/formulas → Find where they map in the network
- **Process**: 
  1. Enumerate candidate algorithms (logic formulas)
  2. Search for neural network locations that implement these algorithms
  3. Validate using Interchange Intervention Accuracy (IIA)
- **Implementation**: `mappings.py` - `find_minimal_mappings()`

### 2. Where-Then-What Strategy  
- **Approach**: Start with network structure → Determine what algorithms it implements
- **Process**:
  1. Find sparse circuits that preserve network functionality
  2. Ground these circuits with specific logic gate interpretations
  3. Validate circuit behavior matches expected algorithms
- **Implementation**: `circuit.py` - `find_circuits()` + `ground()`

## Key Mathematical Concepts

### Circuits
A **circuit** $C$ is a sparse subnetwork defined by:
- **Node masks**: $\mathbf{n}^{(l)} \in \{0,1\}^{d_l}$ for layer $l$
- **Edge masks**: $\mathbf{e}^{(l)} \in \{0,1\}^{d_{l+1} \times d_l}$ for connections between layers

**Constraint**: Circuit must maintain an active path from input to output.

**Sparsity**: $\text{sparsity}(C) = \frac{\text{# masked elements}}{\text{# total elements}}$

### Groundings
A **grounding** $G$ assigns logic gate interpretations to circuit nodes:
$$G: \text{Circuit nodes} \rightarrow \text{Logic gates}$$

**Truth table consistency**: For node $n$ with parents $P(n)$:
$$\forall \mathbf{x} \in \{0,1\}^{|P(n)|}: G(n)(\mathbf{x}) = f_n(G(p_1)(\mathbf{x}), \ldots, G(p_{|P(n)|})(\mathbf{x}))$$

### Mappings  
A **mapping** $M$ connects formula nodes to network components:
$$M: \text{Formula nodes} \rightarrow \mathcal{P}(\text{Network layers} \times \text{Neuron indices})$$

**Minimality**: Mapping $M_1 < M_2$ if $M_1$ is strictly contained in $M_2$.

### Interchange Intervention Accuracy (IIA)
For validation, interventions test if swapping activations preserves behavior:
$$\text{IIA}(f, M, \mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{(\mathbf{x}_1, \mathbf{x}_2) \in \mathcal{D}} \mathbb{I}[f(\mathbf{x}_1^{M \leftarrow \mathbf{x}_2}) = f_{\text{formula}}(\mathbf{x}_1^{M \leftarrow \mathbf{x}_2})]$$

## Code Architecture Analysis

### Core Components

#### `logic_gates.py`
- **LogicGate class**: Represents boolean functions (AND, OR, XOR, etc.)
- **LogicTree class**: Hierarchical representation of logic formulas
- **Truth table operations**: Generate/manipulate boolean function tables
- **Formula generation**: `all_formulas()` creates candidate algorithms systematically

#### `neural_model.py` 
- **MLP class**: Multi-layer perceptron with intervention capabilities
- **Key methods**:
  - `forward()` with circuit masking and interventions
  - `separate_into_k_mlps()` for multi-output analysis
  - `get_states()` for activation extraction

#### `circuit.py`
- **Circuit class**: Sparse network representation with node/edge masks
- **`find_circuits()`**: Enumerates all valid sparse circuits meeting accuracy thresholds
- **`ground()`**: Assigns logic gate interpretations to circuit nodes
- **Validation**: Ensures circuit structure maintains connectivity

#### `grounding.py`
- **Grounding class**: Maps circuit nodes to logic gates
- **Truth table computation**: `enumerate_tts()`, `compute_local_tts()`
- **Consistency checking**: Validates grounding against activations

#### `mappings.py`
- **Mapping class**: Links formula nodes to network locations  
- **`find_mappings()`**: Recursive search for valid node-to-neuron assignments
- **`find_minimal_mappings()`**: Finds non-redundant explanations
- **IIA validation**: Tests intervention consistency

### Experimental Pipeline (`main.py`)

1. **Data Generation**: Create noisy training data for target logic gates
2. **Model Training**: Train MLPs to learn target functions
3. **Circuit Analysis**: Find sparse circuits using where-then-what
4. **Formula Analysis**: Find mappings using what-then-where  
5. **Comparison**: Identify conflicts between explanations

## Key Research Findings (Inferred from Code)

### Non-Identifiability Examples
- **XOR Demo**: Simple 3-layer MLP shows multiple valid explanations
  - Multiple circuits can implement XOR with different structures
  - Multiple formulas can map to same network with different alignments
  
- **MNIST Demo**: Larger networks show extensive non-identifiability
  - 4,702 different circuits found in 3-layer subnetwork
  - Each represents valid but potentially conflicting explanation

### Empirical Observations
- **Formula diversity**: 50+ equivalent logical formulas found for XOR
- **Circuit proliferation**: Even small networks yield many valid circuits
- **Mapping multiplicity**: Single formulas admit multiple neural implementations

## Implications for Mechanistic Interpretability

### Theoretical Impact
1. **Non-uniqueness**: MI explanations are fundamentally non-unique
2. **Strategy dependence**: What-then-where ≠ Where-then-what results
3. **Scale sensitivity**: Problem compounds with network size

### Methodological Concerns  
1. **Cherry-picking risk**: Easy to select preferred explanations
2. **Validation challenges**: Multiple explanations can pass same tests
3. **Reproducibility issues**: Different methods yield different "ground truths"

### Future Directions (Implied)
1. **Consensus methods**: Aggregate across multiple explanation strategies
2. **Probabilistic frameworks**: Assign confidence to competing explanations  
3. **Causal constraints**: Additional structure to constrain explanation space

## Experimental Design Patterns

### Systematic Validation
- **Cross-validation**: Test explanations across multiple random seeds
- **Scale analysis**: Examine how non-identifiability grows with network size
- **Task diversity**: Test across different target functions

### Reproducibility Measures
- **Seed control**: `set_seeds()` for deterministic experiments
- **Hyperparameter sweeps**: Test robustness across training conditions
- **Statistical reporting**: Multiple experiments per configuration

## Technical Implementation Notes

### Performance Optimizations
- **Circuit enumeration**: Uses combinatorial generation with early stopping
- **Grounding computation**: Parallel processing of circuit interpretations
- **Memory management**: Streaming large experiment result sets

### Extensibility Features
- **Custom logic gates**: Easy addition of new target functions
- **Flexible architectures**: Support for various MLP configurations  
- **Visualization tools**: Built-in plotting for all major components

This research provides crucial insights into the fundamental limitations of mechanistic interpretability, demonstrating that the quest for "the true" explanation of neural network behavior may be inherently ill-posed.