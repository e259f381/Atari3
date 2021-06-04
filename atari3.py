# Analysis of the Atati-3 Dataset.

import numpy as np
import pandas as pd
import itertools
import sklearn
import sklearn.linear_model
import statsmodels
import statsmodels.api as sm
import json
import csv
import matplotlib.pyplot as plt
import multiprocessing
import functools
import time
from sklearn.model_selection import cross_val_score

# results are a lot better with an intercept.
USE_INTERCEPT = True

# number of CPU workers to use
PROCESSES = 12

# these are the names of the games in the standard 57-game ALE benchmark.
canonical_57 = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Asteroids",
    "Atlantis",
    "Bank Heist",
    "Battle Zone",
    "Beam Rider",
    "Berzerk",
    "Bowling",
    "Boxing",
    "Breakout",
    "Centipede",
    "Chopper Command",
    "Crazy Climber",
    "Defender",
    "Demon Attack",
    "Double Dunk",
    "Enduro",
    "Fishing Derby",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "Hero",
    "Ice Hockey",
    "James Bond",
    "Kangaroo",
    "Krull",
    "Kung Fu Master",
    "Montezuma Revenge",
    "Ms Pacman",
    "Name This Game",
    "Phoenix",
    "Pitfall",
    "Pong",
    "Private Eye",
    "QBert",
    "Riverraid",
    "Road Runner",
    "Robotank",
    "Seaquest",
    "Skiing",
    "Solaris",
    "Space Invaders",
    "Star Gunner",
    "Surround",
    "Tennis",
    "Time Pilot",
    "Tutankham",
    "Up n Down",
    "Venture",
    "Video Pinball",
    "Wizard of Wor",
    "Yars Revenge",
    "Zaxxon"
]
canonical_57 = [x.lower() for x in canonical_57]

def convert_atari_data(path):
    """
    Extract data from json file (source from https://github.com/paperswithcode/paperswithcode-data)
    Path is to JSON file from paperswithcode
    Saves a CSV file containing results
    """

    with open(path, 'r') as t:
        data = json.load(t)

    atari_key = [i for i in range(len(data)) if 'Atari' in data[i]['task']]
    assert len(atari_key) == 1  # usually item 407, but this might change...
    atari_datasets = data[atari_key[0]]["datasets"]

    games = [atari_datasets[i]['dataset'] for i in range(len(atari_datasets))]
    print(f'Found {len(games)} games')

    def sanitize(s: str):
        return "".join(c for c in s if c not in [','])

    atari_prefix = 'atari 2600 '

    game_map = {
        "montezuma's revenge": "montezuma revenge",
        'q*bert': 'qbert',
        'pitfall!': 'pitfall',
        'kung-fu master': 'kung fu master',
        'ms. pacman': 'ms pacman',
        'river raid': 'riverraid',
        'up and down': 'up n down',
        'pooyan': None,  # these are ignored
        'journey escape': None,
        'elevator action': None,
        'carnival': None,

    }

    with open("Atari-Results.csv", 'w', newline='', encoding='utf-8') as t:
        csv_writer = csv.writer(t, delimiter=',')
        csv_writer.writerow(['Algorithm', 'Score', 'Extra Training Data', 'Paper title', 'Date', 'Game'])

        for dataset in atari_datasets:

            for row in dataset['sota']['rows']:
                if 'Score' not in row['metrics']:
                    continue
                algorithm = row['model_name']
                score = sanitize(row['metrics']['Score'])
                if len(row['metrics']) > 1:
                    print(f"Warning, multiple metrics, for {algorithm} {dataset['dataset']} ignoring non-score. {row['metrics']}")
                extra_training_data = False  # for now
                paper_title = None  # for moment
                paper_date = row['paper_date']
                game = dataset['dataset'].lower()
                if game.startswith(atari_prefix):
                    game = game[len(atari_prefix):]
                if game in game_map:
                    game = game_map[game]
                if game is None:
                    continue
                if game not in canonical_57:
                    print(f" -game {game} ignored")
                    continue
                csv_writer.writerow([algorithm, score, extra_training_data, paper_title, paper_date, game])


def count_57_games(algo):
    """ Returns number of games within the canonical set of 57 that
    this algorithmh has scores for. """
    return 57-len(missing_57_games(algo))

def excess_57_games(algo):
    """ Returns number of games within the canonical set of 57 that
    this algorithmh has scores for. """
    subset = algo_scores[algo_scores["Algorithm"] == algo]
    # filter by 57 games
    subset = subset[np.logical_not(subset["In57"])]
    return subset

def missing_57_games(algo):
    """ Returns all games missing from the Atari-57 datset. """
    subset = algo_scores[algo_scores["Algorithm"] == algo]
    return [game for game in canonical_57 if game not in list(subset["Game"])]

def calculate_median(algo, scores):
    """ Calculate the median score for a given algorithm. """

    # filter by algorithm
    subset = scores[scores["Algorithm"] == algo]

    if len(subset) == 0:
        return float("nan")

    return np.median(subset["Normalized"])

def get_subset(games_set, scores=None):
    """ Returns rows matching any algorithm in games_set"""
    scores = scores if scores is not None else algo_scores
    return scores[[game in games_set for game in scores["Game"]]]


def fit_model(games_subset, algo_scores, true_scores):
    """
    Fit a linear regression model to games subset.
    """
    scores = get_subset(games_subset, algo_scores)
    scores = scores[scores["train"]]

    X_all = scores.pivot_table(index='Algorithm', columns="Game", values="Normalized", fill_value=None)[
        list(games_subset)]
    mask = np.all(X_all.notna(), axis=1)
    X_all = X_all.sort_values("Algorithm")  # needs to be in algorithm order to match true_scores
    y_all = np.asarray([true_scores[algo] for algo in X_all.index])

    X = X_all[mask]
    y = y_all[mask]

    if len(X) < 20:
        print(f"Warning! Too many missing samples for: {games_subset}")

    lm = sklearn.linear_model.LinearRegression(fit_intercept=USE_INTERCEPT)
    lm.fit(X, y)

    return lm, (X, y)

def evaluate_regression_subset(games_subset, algo_scores, true_scores: dict, verbose=False):
    """
    True scores must be in sorted order according to algos
    """

    lm, (X, y) = fit_model(games_subset, algo_scores, true_scores)

    predicted_scores = lm.predict(X)
    errors = np.abs(y - predicted_scores)
    rms = (errors ** 2).mean() ** 0.5
    r2 = lm.score(X, y)

    SS_Residual = sum((y - predicted_scores) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total

    N = len(y)
    K = X.shape[1]
    adjusted_r2 = 1 - (1 - r_squared) * (N - 1) / (N - K - 1)

    _coef = [lm.intercept_] + [lm.coef_]

    if verbose:
        import statsmodels
        import statsmodels.api as sm
        mod = sm.OLS(y, X)
        fii = mod.fit()
        p_values = fii.summary2().tables[1]['P>|t|']
        # plt.scatter(X,y,marker='x',label="True")
        # plt.scatter(X,predicted_scores,label="Pred")
        # plt.show()
        print("Intercept: {}, coef: {} r^2: {} p:{}".format(lm.intercept_, lm.coef_, lm.score(X, y), p_values))
        print("{:<30} {:<10} {:<10} {:<10}".format("algo", "true", "estimated", "error"))
        for algo, true_score, subset_score, error in zip(algos, true_scores, predicted_scores, errors):
            print("{:<30} {:<10.1f} {:<10.1f} {:<10.1f} {}".format(algo, true_score, subset_score, error, _coef))

    return (1-adjusted_r2, 1-r2, rms, games_subset, _coef)


def search_regression(r=3, always_envs=None, banned_envs=None, verbose=True, top_k=57):
    """
    Search over subsets.
    """

    if banned_envs is None:
        banned_envs = []

    if always_envs is None:
        always_envs = tuple()

    best = None
    counter = 0
    print("Checking all sets of {} games.".format(r))

    algos = sorted(good_algos)
    true_scores = {k: median_scores[k] for k in algos}

    filtered_scores = algo_scores[[algo in good_algos for algo in algo_scores["Algorithm"]]]

    if r >= 2:
        # these games have less than 20 samples and so we ignore them.
        # it seems that name-this-game is a good predictor, as is yars revenge... but like I say there just
        # aren't enough datapoints for these to be useful, especially when looking at old data.
        banned_games = ['defender', 'phoenix', 'pitfall', 'skiing', 'solaris', 'surround', 'yars revenge'] + banned_envs
    else:
        banned_games = [] + banned_envs
    games_with_enough_data = [game for game in canonical_57 if game.lower() not in banned_games]

    combinations = list(itertools.combinations(games_with_enough_data, r-len(always_envs)))

    # add always envs in...
    combinations = [x+always_envs for x in combinations]

    start_time = time.time()

    if PROCESSES > 1:
        with multiprocessing.Pool(processes=PROCESSES) as pool:
            results = pool.map(
                functools.partial(evaluate_regression_subset, algo_scores=filtered_scores, true_scores=true_scores),
                combinations
            )
    else:
        results = list(map(
            functools.partial(evaluate_regression_subset, algo_scores=filtered_scores, true_scores=true_scores),
            combinations
        ))

    time_taken = time.time() - start_time
    fps = len(results) / time_taken

    print(f"Generated {len(results)} models in {time_taken:.1f}s at {fps:.1f} models/second.")

    results.sort(reverse=True)

    for (inv_r2_adj, inv_r2, rms, games_subset, _coef) in results[-top_k:]:
        if verbose:
            print(f"{str(games_subset):<60} {1-inv_r2_adj:<10.5f} {1-inv_r2:<10.5f} {100*inv_r2_adj:<10.5f} {100*inv_r2:<10.5f} {rms:<10.1f} {_coef}")
        counter += 1

    return results


def run_init(do_not_train_on=None, verbose=False):
    """
    Run initialization.
    """

    do_not_train_on = do_not_train_on or []

    global good_algos
    global algo_scores
    global median_scores
    global algos
    global human_scores

    if verbose:
        print("Number of canonical games {}".format(len(canonical_57)))

    pd.set_option('display.max_columns', None)

    convert_atari_data('evaluation-tables.json')

    human_scores = pd.read_csv("Atari-Human.csv", dtype={"Score": float})

    # I added some supplementary results to fill in missing data...
    algo_scores1 = pd.read_csv("Atari-Results.csv", dtype={"Score": float})
    algo_scores2 = pd.read_csv("Atari-Sup.csv", dtype={"Score": float})
    algo_scores = pd.concat([algo_scores1, algo_scores2])

    # make all names lower case
    human_scores["Game"] = human_scores["Game"].str.lower()
    algo_scores["Game"] = algo_scores["Game"].str.lower()

    for index, row in algo_scores.iterrows():
        if row["Game"] not in canonical_57:
            print(f"Invalid game {row['Game']} on algorithm {row['Algorithm']}")

    algo_scores = algo_scores.merge(human_scores[["Game", "Random", "Human"]], on="Game", how="left")
    algo_scores["Normalized"] = 100 * (algo_scores["Score"] - algo_scores["Random"]) / (
            algo_scores["Human"] - algo_scores["Random"])

    algo_scores["In57"] = [game in canonical_57 for game in algo_scores["Game"]]

    all_algorithms_list = set(algo_scores["Algorithm"])
    if verbose:
        print("All algorithms:", all_algorithms_list)

    for game in do_not_train_on:
        assert game in all_algorithms_list, f"{game} missing from algorithms list"

    algo_scores["train"] = [game not in do_not_train_on for game in algo_scores["Algorithm"]]

    algorithms = set(algo_scores["Algorithm"])
    good_algos = [algo for algo in sorted(algorithms) if count_57_games(algo) >= 40]
    print(f"Algorithms total {len(algorithms)}, algorithms with 40 or more games {len(good_algos)}")
    bad_algos = [algo for algo in sorted(algorithms) if algo not in good_algos]
    if verbose:
        print("Good:", good_algos)
        print("Bad:", bad_algos)

    # median before algorithm filter...
    median_scores = {
        k: calculate_median(k, get_subset(canonical_57)) for k in good_algos
    }
    all_median_scores = {
        k: calculate_median(k, get_subset(canonical_57)) for k in algorithms
    }

    # filter bad algorithms
    algo_scores["good"] = [algo in good_algos for algo in algo_scores["Algorithm"]]
    algo_scores = algo_scores[algo_scores['good']]
    algo_scores["Game"] = algo_scores["Game"].astype('category')  # faster?
    median_sorted_scores = sorted([(v, k) for k, v in all_median_scores.items()])

    if verbose:
        print(f"Found {len(good_algos)} datapoints with 40 or more games.")
        for n_games in reversed(range(1, 57 + 1)):
            matching = [algo for algo in algorithms if count_57_games(algo) == n_games]
            if len(matching) > 0:
                print(f"[{n_games:02d}] {matching}")

        print()
        print("Missing games:")
        for algo in good_algos:
            print(f" -{algo}: {missing_57_games(algo)}")


        print()
        print("Median_57 scores:")

        for score, algo in median_sorted_scores:
            if algo not in good_algos:
                continue
            marker = "*" if algo in do_not_train_on else ""
            if algo not in good_algos:
                marker = marker + " -"
            print(f" -{algo}: {score:.0f} {marker}")


if __name__ == "__main__":

    # ---------------------------------------------------
    # find a good subsets...

    run_init(do_not_train_on=["FirstReturn"])

    search_regression(1)
    search_regression(2, top_k=10)
    search_regression(3, top_k=10)

    atari3 = ('battle zone', 'gopher', 'time pilot')
    search_regression(3, banned_envs=list(atari3), top_k=10)
    search_regression(5, always_envs=atari3, top_k=10)

    search_regression(4, top_k=10)
    search_regression(5, top_k=10)

