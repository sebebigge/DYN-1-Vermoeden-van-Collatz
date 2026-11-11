"""
@author: Sebastian Bigge
@date:   2026-11-09
"""

# Imports
import os
import time
import numpy as np
import matplotlib.pyplot as plt


####################################################################################
################################  IMPLEMENTATION  ##################################
####################################################################################

"""
"""
def brute_force(n: int) -> int:
    if n <= 1:
        return 0
    n = n // 2 if n % 2 == 0 else 3 * n + 1
    return 1 + brute_force(n)


"""
"""
def brute_force_list(max_n: int) -> tuple[list[int], int]:
    steps = [0] * (max_n + 1)
    for n in range(1, max_n + 1):
        steps[n] = brute_force(n)
    return steps, sum(steps)


"""
"""
def top_down(n: int, steps: dict, counter: list[int]) -> int:
    if n in steps:
        return steps[n]
    counter[0] += 1
    next_n = n // 2 if n % 2 == 0 else 3 * n + 1
    steps[n] = 1 + top_down(next_n, steps, counter)
    return steps[n]


"""
"""
def top_down_list(max_n: int) -> tuple[list[int], int]:
    counter = [0]
    steps = {1: 0}
    result = [0] * (max_n + 1)
    for i in range(2, max_n + 1):
        result[i] = top_down(i, steps, counter)
    return result, counter[0]


"""
"""
def bottom_up_list(max_n: int) -> tuple[list[int], int]:
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

# Colour-blind friendly (Okabe-Ito) + distinct markers/line styles (readable in B/W)
STYLE = {
    "brute force": dict(color="#D55E00", marker="o", ls="-"),
    "top-down":    dict(color="#0072B2", marker="s", ls="--"),
    "bottom-up":   dict(color="#009E73", marker="^", ls=":"),
}
FIGSIZE = (5.5, 3.6)
XLABEL_MAX = r"Input size $n_{\max}$"


def _new_figure():
    return plt.subplots(figsize=FIGSIZE, constrained_layout=True)


def _lines_vs_max(limits, data, ylabel, logy=True):
    """One line per method in `data` against MAX (log x-axis)."""
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


"""
Collatz steps for every start value n (identical for all methods).
Plots at most `max_points` points so huge MAX values stay fast.
"""
def plot_steps_vs_n(steps, max_points=200_000):
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


"""
Total number of computed steps vs MAX (log-log).
Brute force appears only if it was run (it is then a key of `operations`).
"""
def plot_operations(limits, operations):
    fig, _ = _lines_vs_max(limits, operations, "Number of computed steps")
    return fig


"""
Average number of steps per number (ops / MAX) vs MAX.
"""
def plot_steps_per_number(limits, operations):
    per_n = {name: np.array(ops) / np.array(limits) for name, ops in operations.items()}
    fig, _ = _lines_vs_max(
        limits, per_n,
        r"Average steps per number ($\mathrm{ops}/n_{\max}$)", logy=False)
    return fig


"""
Computation time vs MAX (log-log).
"""
def plot_times(limits, times, runs):
    fig, _ = _lines_vs_max(limits, times, f"Computation time (s, best of {runs})")
    return fig


"""
Efficiency gain: share of the brute-force steps that is saved by reuse (%).
The brute-force step count of MAX equals sum(steps[1..MAX]), so it is derived from
the step table and works whether or not brute force was actually run.
"""
def plot_saved_share(limits, operations, steps):
    cumulative = np.cumsum(steps)                       # cumulative[n] = sum of steps[0..n]
    brute_force_ops = cumulative[np.array(limits)]
    shares = {name: 100 * (1 - np.array(ops) / brute_force_ops)
              for name, ops in operations.items() if name != "brute force"}
    fig, ax = _lines_vs_max(limits, shares, "Share of steps saved (%)", logy=False)
    ax.set_ylim(0, 100)
    return fig


def save_figure(fig, name, folder=OUT):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, f"{name}.png"))


"""
Efficiency gain in time: share of the brute-force time that is saved (%).
Unlike the step count, the brute-force time cannot be derived, so brute force
must have been run. Negative values mean slower than brute force.
"""
def plot_time_saved(limits, times, ymin=None):
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


"""
Draw all five plots, each in its own figure.
"""
def plot_all(results, show: bool = True, save_figs: bool = True, folder=None):
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


"""
"""
def timer(function, max_n):
    t = time.perf_counter()
    steps, counter = function(max_n)
    return steps, counter, time.perf_counter() - t


"""
Run the given methods for several values of MAX, collect times/operations/steps
and plot the results. Used by test_main and test_bottom_up.
"""
def run_benchmark(methods: dict, max_n: int, runs: int, nbPoints: int, min_n: int,
                  show: bool, save_figs: bool, folder):
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
    if "brute force" in methods:    # brute-force operations must equal sum(steps[1..MAX])
        cumulative = np.cumsum(steps)
        assert all(o == cumulative[l] for o, l in zip(operations["brute force"], limits))
    results = {"limits": limits, "times": times, "operations": operations,
               "steps": steps, "runs": runs}
    plot_all(results, show=show, save_figs=save_figs, folder=folder)
    return results


"""
Brute force is only run when include_brute_force=True (it is far slower).
"""
def test_main(max_n: int, runs: int = 5, nbPoints: int = 10, min_n: int = 2,
              include_brute_force: bool = False, show: bool = True,
              save_figs: bool = True, folder=None):
    methods = {name: f for name, f in METHODS.items()
               if include_brute_force or name != "brute force"}
    return run_benchmark(methods, max_n, runs, nbPoints, min_n, show, save_figs, folder)


"""
Run only the bottom-up method. Figures go to their own folder so they do
not overwrite those of test_main.
"""
def test_bottom_up(max_n: int, runs: int = 5, nbPoints: int = 10, min_n: int = 2,
                   show: bool = True, save_figs: bool = True, folder=None):
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