"""
@author: Sebastian Bigge
@date:   2026-11-11
"""

# Imports
import os
import time
import numpy as np
import matplotlib.pyplot as plt


####################################################################################
################################  IMPLEMENTATION  ##################################
####################################################################################


def brute_force(n: int) -> int:
    """Return the number of Collatz steps required to reach 1 from ``n``.

    The function follows the standard Collatz recurrence:

    * if ``n`` is even, set ``n = n // 2``;
    * if ``n`` is odd, set ``n = 3 * n + 1``.

    The recursion stops when ``n`` becomes 1, at which point the sequence length is
    defined as zero additional steps. This implementation is intentionally simple and
    therefore very slow for large inputs, making it useful as a baseline for correctness
    and performance comparisons.

    Args:
        n: The starting value of the Collatz sequence.

    Returns:
        The number of transitions needed until the sequence reaches 1.
    """
    if n <= 1:
        return 0
    n = n // 2 if n % 2 == 0 else 3 * n + 1
    return 1 + brute_force(n)


def brute_force_list(max_n: int) -> tuple[list[int], int]:
    """Construct the full step table for all integers ``1..max_n`` with brute force.

    Each entry ``steps[i]`` contains the Collatz length of ``i``. The function returns
    both the complete list and the total number of steps across all starting values,
    which is then used as a baseline count in the benchmark plots.

    Args:
        max_n: Largest starting value to include in the table.

    Returns:
        A tuple ``(steps, total_steps)`` where ``steps[i]`` is the number of Collatz
        transitions needed for ``i`` and ``total_steps`` is the sum over all entries.
    """
    steps = [0] * (max_n + 1)
    for n in range(1, max_n + 1):
        steps[n] = brute_force(n)
    return steps, sum(steps)


def top_down(n: int, steps: dict, counter: list[int]) -> int:
    """Compute the Collatz length of ``n`` using memoised recursion.

    The function stores known solutions in ``steps`` so every value is solved at most once.
    The ``counter`` list tracks how many memoised recursive subproblems were expanded, which
    is useful for comparing the amount of work done by this strategy to the naive approach.

    Args:
        n: The starting value whose Collatz length should be computed.
        steps: Mapping from already-solved values to their step counts.
        counter: A one-element list used to count the number of newly computed states.

    Returns:
        The number of Collatz steps required to reach 1 from ``n``.
    """
    if n in steps:
        return steps[n]
    counter[0] += 1
    next_n = n // 2 if n % 2 == 0 else 3 * n + 1
    steps[n] = 1 + top_down(next_n, steps, counter)
    return steps[n]


def top_down_list(max_n: int) -> tuple[list[int], int]:
    """Build a memoised Collatz table for all values ``1..max_n``.

    The algorithm initializes the base state ``steps[1] = 0`` and computes each value from
    2 upward. Values already present in the memoisation dictionary are reused, so the method
    avoids repeating work across sequences that merge at common states.

    Args:
        max_n: Largest starting value included in the table.

    Returns:
        A tuple ``(result, computed_states)`` where ``result[i]`` is the step count for
        ``i`` and ``computed_states`` counts how many intermediate values were expanded.
    """
    counter = [0]
    steps = {1: 0}
    result = [0] * (max_n + 1)
    for i in range(2, max_n + 1):
        result[i] = top_down(i, steps, counter)
    return result, counter[0]


def bottom_up_list(max_n: int) -> tuple[list[int], int]:
    """Compute the entire Collatz step table in increasing order using dynamic programming.

    For even ``i``, the function reuses the already-known value for ``i // 2`` and adds one
    step. For odd ``i``, it follows the next terms until it reaches a value that is known to
    be already computed; this produces a dynamic-programming recurrence that works well when
    many values share the same future trajectory. This method is often much faster than the
    naive recursive strategy while remaining conceptually simple.

    Args:
        max_n: Largest starting value included in the table.

    Returns:
        A tuple ``(steps, counter)`` where ``steps[i]`` is the Collatz length for ``i`` and
        ``counter`` is the number of update steps performed in the DP loop.
    """
    counter = 0
    steps = [0] * (max_n + 1)
    for i in range(2, max_n + 1):
        if i % 2 == 0:
            steps[i] = steps[i // 2] + 1
            counter += 1
        else:
            j, c = i, 0
            while j >= i:
                j = j // 2 if j % 2 == 0 else 3 * j + 1
                c += 1
            steps[i] = c + steps[j]
            counter += c
    return steps, counter


####################################################################################
####################################  PLOTS  #######################################
####################################################################################

OUT = "Graphs"

# GenAI
# Style for an academic report (serif, light grid, 300 dpi PNG export)
STYLE_RC = {
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "mathtext.fontset": "cm",
}

# GenAI
# Colour-blind friendly (Okabe-Ito) + distinct markers/line styles (readable in B/W)
STYLE = {
    "brute force": dict(color="#D55E00", marker="o", ls="-"),
    "top-down":    dict(color="#0072B2", marker="s", ls="--"),
    "bottom-up":   dict(color="#009E73", marker="^", ls=":"),
}
FIGSIZE = (5.5, 3.6)
XLABEL_MAX = r"Input size $n_{\max}$"


# GenAI
def _new_figure():
    """Create a fresh Matplotlib figure configured for the project’s report style.

    The helper centralises the default figure size and layout so every plotting function
    starts from a consistent canvas. It keeps the visualisation compact and publication-
    friendly, which is helpful when many benchmark curves are generated for a report.

    Returns:
        A ``(fig, ax)`` tuple produced by Matplotlib using the project's chosen dimensions
        and a constrained layout.
    """
    return plt.subplots(figsize=FIGSIZE, constrained_layout=True)


# GenAI
def _lines_vs_max(limits, data, ylabel, logy=True):
    """Plot one line per method in ``data`` against the maximum input bound.

    This helper is used for benchmark comparisons such as computation cost, time, and
    relative savings. It places all curves on a log-scaled x-axis by default and optionally
    also applies a log scale on the y-axis when the metric is naturally multiplicative.

    Args:
        limits: Sequence of upper bounds used on the x-axis.
        data: Mapping from method name to a sequence of values to plot.
        ylabel: Axis label describing the metric being visualised.
        logy: Whether to use a logarithmic y-axis for the plotted values.

    Returns:
        A tuple ``(fig, ax)`` containing the created figure and its axes.
    """
    fig, ax = _new_figure()
    for name, values in data.items():
        ax.plot(limits, values, label=name, markersize=4, lw=1.3, **STYLE[name])
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(XLABEL_MAX)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    return fig, ax


# GenAI
def plot_steps_vs_n(steps, max_points=200_000):
    """Scatter the Collatz step count for each starting value in the computed table.

    The function samples the step data to keep visualisation responsive for very large
    ranges. It annotates the value with the largest step count so the plot highlights the
    most expensive starting value in the analysed interval.

    Args:
        steps: A sequence indexed by starting value whose entry is the number of Collatz
            steps until reaching 1.
        max_points: Maximum number of points to render; larger tables are downsampled.

    Returns:
        The matplotlib figure containing the scatter plot.
    """
    fig, ax = _new_figure()
    stride = max(1, (len(steps) - 1) // max_points)
    xs = range(1, len(steps), stride)
    ax.scatter(xs, [steps[n] for n in xs], s=2, alpha=0.5,
               color="#555555", rasterized=True)
    longest = max(range(1, len(steps)), key=steps.__getitem__)
    ax.annotate(f"n={longest}: {steps[longest]} steps",
                (longest, steps[longest]), xytext=(0.25, 0.92),
                textcoords="axes fraction", arrowprops={"arrowstyle": "->"})
    ax.set_xlabel(r"Start value $n$")
    ax.set_ylabel("Steps to reach 1")
    return fig


# GenAI
def plot_operations(limits, operations):
    """Plot the total number of computed Collatz states against the input bound.

    Each method contributes a line whose y-value is the number of states that had to be
    computed or expanded during the experiment. This plot shows the raw amount of work,
    independent of the actual wall-clock time.

    Args:
        limits: Sequence of maximum input sizes tested.
        operations: Mapping from method name to a sequence of operation counts.

    Returns:
        The generated matplotlib figure.
    """
    fig, _ = _lines_vs_max(limits, operations, "Number of computed steps")
    return fig


# GenAI
def plot_steps_per_number(limits, operations):
    """Plot the average Collatz steps per starting value for each method.

    This metric is computed as ``operations / n_max`` and is useful for comparing how much
    work each approach does per input value as the problem size grows. Unlike the raw total
    work, it normalises the values by the maximum start value that was analyzed.

    Args:
        limits: Upper bounds used as divisors in the average calculation.
        operations: Mapping from method name to total operation counts.

    Returns:
        The generated matplotlib figure.
    """
    per_n = {name: np.array(ops) / np.array(limits) for name, ops in operations.items()}
    fig, _ = _lines_vs_max(
        limits, per_n,
        r"Average steps per number ($\mathrm{ops}/n_{\max}$)", logy=False)
    return fig


# GenAI
def plot_times(limits, times, runs):
    """Plot the best observed runtime for each method across several benchmark limits.

    Each benchmark stores the minimum runtime over a set of repeated runs, which reduces the
    effect of transient system noise. The resulting figure highlights how each algorithm
    scales as the input size increases.

    Args:
        limits: Maximum input sizes used for each measurement.
        times: Mapping from method name to the minimum runtime observed for each limit.
        runs: Number of runs used to determine the best time at each limit.

    Returns:
        The generated matplotlib figure.
    """
    fig, _ = _lines_vs_max(limits, times, f"Computation time (s, best of {runs})")
    return fig


# GenAI
def plot_saved_share(limits, operations, steps):
    """Plot the percentage of brute-force work saved by reusable dynamic programming.

    The calculation compares the operation counts of each method to the cumulative total of
    the brute-force step table, which is equivalent to the total work of a naive solution.
    This yields a direct measure of how much redundant computation is avoided by the caching
    and DP approaches.

    Args:
        limits: Sequence of upper bounds used in the analysis.
        operations: Mapping from method name to total operation counts.
        steps: A step table whose cumulative sum yields the brute-force baseline.

    Returns:
        The generated matplotlib figure.
    """
    cumulative = np.cumsum(steps)                       # cumulative[n] = sum of steps[0..n]
    brute_force_ops = cumulative[np.array(limits)]
    shares = {name: 100 * (1 - np.array(ops) / brute_force_ops)
              for name, ops in operations.items() if name != "brute force"}
    fig, ax = _lines_vs_max(limits, shares, "Share of steps saved (%)", logy=False)
    ax.set_ylim(0, 100)
    return fig


# GenAI
def save_figure(fig, name, folder=OUT):
    """Save a Matplotlib figure to disk under the project's graph directory.

    The helper creates the output folder if needed and writes a PNG file using the
    requested identifier as the filename. This keeps the plotting utilities straightforward
    to reuse in repeated benchmark runs.

    Args:
        fig: Matplotlib figure to save.
        name: Base filename for the image, without the extension.
        folder: Directory to store the output image in.
    """
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, f"{name}.png"))


# GenAI
def plot_time_saved(limits, times, ymin=None):
    """Plot the percentage of brute-force runtime saved by each method.

    This metric compares each method's runtime against the brute-force baseline. A value of
    50% indicates that the method is twice as fast as the cubic baseline. Negative values mean
    a method is slower than brute force.

    Args:
        limits: Sequence of benchmark upper bounds.
        times: Mapping from method name to runtime series measured for each limit.
        ymin: Optional lower bound used to clamp the y-axis when the range is too broad.

    Returns:
        The generated matplotlib figure.

    Raises:
        ValueError: If the brute-force timings are missing because they are required to
            compute the relative time savings.
    """
    if "brute force" not in times:
        raise ValueError("plot_time_saved needs brute-force timings (include_brute_force=True)")
    brute = np.array(times["brute force"])
    with np.errstate(divide="ignore", invalid="ignore"):
        saved = {name: 100 * (1 - np.array(t) / brute)
                 for name, t in times.items() if name != "brute force"}
    fig, ax = _lines_vs_max(limits, saved, "Time saved (%)", logy=False)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylim(top=100)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    return fig


# GenAI
def plot_all(results, show: bool = True, save_figs: bool = True, folder=None):
    """Generate the full collection of benchmark plots for a completed experiment.

    The helper expects a result dictionary produced by ``run_benchmark`` and creates the
    figures necessary to compare total work, average work, timing, and efficiency gains. It
    can either display the plots immediately, save them to disk, or do both.

    Args:
        results: Dictionary containing the benchmark ``limits``, ``times``, ``operations``,
            ``steps`` and ``runs`` metadata.
        show: If ``True``, display the generated figures in a Matplotlib window.
        save_figs: If ``True``, export each figure as a PNG file.
        folder: Optional directory used when saving. Defaults to ``Graphs/N{max(limit)}``.

    Returns:
        A dictionary of figure objects keyed by the names used for each output plot.
    """
    limits, steps = results["limits"], results["steps"]
    if folder is None:
        folder = os.path.join(OUT, f"N{max(limits)}")
    with plt.rc_context(STYLE_RC):
        """figs = {
            "fig1_total_steps": plot_operations(limits, results["operations"]),
            "fig2_steps_per_number": plot_steps_per_number(limits, results["operations"]),
            "fig3_time": plot_times(limits, results["times"], results["runs"]),
            "fig4_share_saved": plot_saved_share(limits, results["operations"], steps),
            "fig5_steps_per_start_value": plot_steps_vs_n(steps),
        }"""
        figs = {"fig6_time_saved": plot_time_saved(limits, results["times"])}
        if save_figs:
            for name, fig in figs.items():
                save_figure(fig, name, folder)
    if show:
        plt.show()
    else:
        plt.close("all")
    return figs


####################################################################################
####################################  TESTS  #######################################
####################################################################################

METHODS = {
    "brute force": brute_force_list,
    "top-down": top_down_list,
    "bottom-up": bottom_up_list,
}


def timer(function, max_n):
    """Execute one benchmark call and measure its wall-clock time.

    This helper wraps a method such as ``brute_force_list`` or ``top_down_list`` so the
    benchmark loop can repeatedly time the same algorithm on a range of input sizes. It
    records both the output data and the elapsed time, which are then used to compare
    algorithmic efficiency.

    Args:
        function: Callable that accepts a maximum value and returns ``(steps, counter)``.
        max_n: Upper bound passed to the benchmarked function.

    Returns:
        A tuple ``(steps, counter, elapsed_seconds)`` containing the solution table,
        the method-specific counter, and the elapsed wall-clock time.
    """
    t = time.perf_counter()
    steps, counter = function(max_n)
    return steps, counter, time.perf_counter() - t


def run_benchmark(methods: dict, max_n: int, runs: int, nbPoints: int, min_n: int,
                  show: bool, save_figs: bool, folder):
    """Benchmark a set of functions on a logarithmically spaced range of input bounds.

    The method runs each algorithm at several sizes between ``min_n`` and ``max_n`` and
    records the best runtime over repeated trials. It also verifies that all methods agree
    on the computed Collatz step table at each tested limit, using the first method's result
    as the reference. The collected data is then passed into the plotting helpers to produce
    the final comparison figures.

    Args:
        methods: Dictionary mapping method names to callable implementations.
        max_n: Largest upper bound to use in the benchmark.
        runs: Number of repeated timing runs for each method and size.
        nbPoints: Number of sample sizes to include in the benchmark grid.
        min_n: Smallest upper bound included in the benchmark grid.
        show: If ``True``, display the resulting figures after generation.
        save_figs: If ``True``, save the plots to the supplied output folder.
        folder: Directory in which the figures will be stored.

    Returns:
        A dictionary containing the benchmark limits, timings, operation counts, step table,
        and the number of runs used for each size.
    """
    limits = [int(l) for l in
              np.unique(np.round(np.logspace(np.log10(min_n), np.log10(max_n), nbPoints)).astype(int))]
    times = {name: [] for name in methods}
    operations = {name: [] for name in methods}
    steps = None
    print(f"{'MAX':>10} | {'method':<11} | {'time (s)':>9} | {'operations':>13}")
    for limit in limits:
        reference, reference_name = None, None
        for name, f in methods.items():
            best = float("inf")
            for _ in range(runs):
                result, ops, t = timer(f, limit)
                best = min(best, t)
            times[name].append(best)
            operations[name].append(ops)
            if reference is None:
                reference, reference_name = result, name
            assert result == reference, f"{name} disagrees with {reference_name}!"
            print(f"{limit:>10} | {name:<11} | {best:>9.4f} | {ops:>13,}")
        print("-" * 52)
        steps = reference
    if "brute force" in methods:
        cumulative = np.cumsum(steps)
        assert all(o == cumulative[l] for o, l in zip(operations["brute force"], limits))
    results = {"limits": limits, "times": times, "operations": operations,
               "steps": steps, "runs": runs}
    plot_all(results, show=show, save_figs=save_figs, folder=folder)
    return results


def test_main(max_n: int, runs: int = 5, nbPoints: int = 10, min_n: int = 2,
              include_brute_force: bool = False, show: bool = True,
              save_figs: bool = True, folder=None):
    """Benchmark the main comparison configuration used for report plots.

    By default the brute-force method is omitted to keep experiments computationally
    feasible at large input sizes. When requested, it can be included to provide a baseline
    against which the more efficient strategies are measured.

    Args:
        max_n: Largest starting value included in the benchmark range.
        runs: Number of repeated timing runs per benchmark point.
        nbPoints: Number of logarithmically spaced sample sizes.
        min_n: Smallest benchmark maximum size.
        include_brute_force: If ``True``, include the naive algorithm in the benchmark.
        show: Whether to open the generated figures interactively.
        save_figs: If ``True``, save the plots to the output directory.
        folder: Optional custom folder for saved figures.

    Returns:
        The benchmark result dictionary produced by ``run_benchmark``.
    """
    methods = {name: f for name, f in METHODS.items()
               if include_brute_force or name != "brute force"}
    return run_benchmark(methods, max_n, runs, nbPoints, min_n, show, save_figs, folder)


def test_bottom_up(max_n: int, runs: int = 5, nbPoints: int = 10, min_n: int = 2,
                   show: bool = True, save_figs: bool = True, folder=None):
    """Benchmark only the bottom-up dynamic-programming strategy.

    A dedicated output folder is created by default so the figures do not overwrite those
    generated by the general benchmark suite. This is useful when comparing the bottom-up
    method separately in a focused analysis.

    Args:
        max_n: Largest starting value included in the benchmark range.
        runs: Number of repeated timing runs per benchmark point.
        nbPoints: Number of logarithmically spaced sample sizes.
        min_n: Smallest benchmark maximum size.
        show: Whether to open the generated figures interactively.
        save_figs: If ``True``, save the plots to disk.
        folder: Optional custom folder for output images; a default bottom-up folder is used
            when omitted.

    Returns:
        The benchmark result dictionary produced by ``run_benchmark``.
    """
    if folder is None:
        folder = os.path.join(OUT, f"N{max_n}_bottom_up")
    methods = {"bottom-up": bottom_up_list}
    return run_benchmark(methods, max_n, runs, nbPoints, min_n, show, save_figs, folder)


if __name__ == "__main__":
    for size in [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
        start_ = time.perf_counter()
        test_main(size, show=False, include_brute_force=True)
        print(f"Total: {time.perf_counter() - start_:.1f} s")


"""
Feasable:
test_main(1_000_000, 10, 10)
test_main(1_000_000, 100, 10)
keep runs at 5 and points at 10
max size for brute force is 10_000_000 (takes about 13 min)
-> each new size time is times 5 (times 5 for the runs)
"""