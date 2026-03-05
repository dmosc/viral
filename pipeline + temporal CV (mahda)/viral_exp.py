import pprint
# Data munging libraries
import numpy as np
import pandas as pd

# For grid searching iteration
import itertools
import uuid
from math import prod

# Utility modules
import yaml # type: ignore
import sys
import os
import pickle
import logging as lg
import warnings
from datetime import datetime
from scipy.stats import zscore

def import_data() -> pd.DataFrame:
    '''
    TBI. This function is for data-preprocessing if we need to do any of it
    '''
    
    #import data
    flat_data = pd.read_csv('')
    #preprocessing stuff
    return flat_data
    

# Helper Funcs.
def parse_config(path: str) -> tuple[dict, dict]:
    '''
    parse_config Reads the YAML config file for this experiment, converts
        relevant params to the correct datetime formats, then returns it as a
        single dict

    Parameters
    ----------
    path : str
        A string containing the filepath, assumes anchor is working directory

    Returns
    -------
    exp_config : dict
        Contains all the model and experiment parameters that are iterated
        *within* run_experiment()
    temporal_config : dict
        Contains all experimental parameters for which *each combination*
        configures an iteration of run_experiment()
    '''
    
    #read in experiment parameter dicts from config YAML
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    #initialize returned dicts
    exp_config, temporal_config = dict(), config['temporal_grid']
    exp_config['model_grid'] = config['model_grid']
    
    #convert temporal config elements to pandas datetime formats
    temporal_config['label_start'] = [
        pd.to_datetime(d) for d in config['temporal_grid']['label_start']
    ]
    temporal_config['label_end'] = [
        pd.to_datetime(d) for d in config['temporal_grid']['label_end']
    ]
    temporal_config['feature_timeframe_months'] = [
        pd.DateOffset(months = m)
        for m in temporal_config['feature_timeframe_months']
    ]
    temporal_config['model_update_freq_months'] = [
        pd.DateOffset(months = m)
        for m in temporal_config['model_update_freq_months']
    ]
    temporal_config['max_training_history_years'] = [
        pd.DateOffset(years = y)
        for y in temporal_config['max_training_history_years']
    ]
    
    #add the config YAML's path to the experiment dict along w targets
    exp_config['path'] = path
    exp_config['targets'] = config['experiment_grid']['targets']
    exp_config['seed'] = config['seed']
    
    #add the countries list to the temporal config
    exp_config["entity_filter"] = config["experiment_grid"].get("entities", None)
    
    #return config dicts
    return exp_config, temporal_config

def month_diff(x: pd.Timestamp, y: pd.Timestamp) -> int:
    return (y.year - x.year) * 12 + (y.month - x.month)

def ensure_monthly_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out

def safe_drop(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

# Dataset adapter: build (X, y, meta) for a list of label dates
def build_supervised_matrix(
    data: pd.DataFrame,
    label_dates: list[pd.Timestamp],
    temporal_spec: dict,
    *,
    entity_col: str,
    date_col: str,
    target_col: str,
    feature_cols: list[str] | None = None,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    For each label date d in label_dates:
      - take rows whose month_d is in [h, h+feature_window)
      - pivot features by lag (month_d) to wide columns: f"{feat}_lag{month_d}"
      - one row per entity (per label_date)
      - attach y = target_col at date == d
    Returns:
      X: features (wide)
      y: target
      meta: entity + label_date (for merging preds later)
    """

    h = temporal_spec["h"]
    feature_window = temporal_spec["feature_timeframe_months"].months  # matches your usage :contentReference[oaicite:5]{index=5}

    if drop_cols is None:
        drop_cols = []
    if feature_cols is None:
        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        # I need to keep an eye on this if the user_id is int
        feature_cols = [c for c in numeric_cols if c not in {target_col}]
    else:
        # Make sure they exist
        feature_cols = [c for c in feature_cols if c in data.columns]

    # We only need these columns to build X/y
    keep = [entity_col, date_col, target_col] + feature_cols
    df = data[keep].copy()
    df = safe_drop(df, drop_cols)

    X_rows = []
    y_rows = []
    meta_rows = []

    y_lookup = df[[entity_col, date_col, target_col]].dropna()
    y_lookup = y_lookup.set_index([entity_col, date_col])[target_col]

    for d in label_dates:
        tmp = df.copy()
        tmp["month_d"] = tmp[date_col].apply(lambda x: month_diff(x, d))

        long_X = tmp[
            (tmp["month_d"] >= h) &
            (tmp["month_d"] < (h + feature_window))
        ].copy()

        # add label_date to keep rows distinct when concatenating across d
        long_X["label_date"] = d

        # Pivot: index = (entity, label_date); columns = (feature, lag)
        wide = long_X.pivot_table(
            index=[entity_col, "label_date"],
            columns="month_d",
            values=feature_cols,
            aggfunc="first"  # same spirit as your code :contentReference[oaicite:6]{index=6}
        )

        wide.columns = [f"{feat}_lag{lag}" for feat, lag in wide.columns]
        wide = wide.reset_index()


        y = wide.apply(lambda r: y_lookup.get((r[entity_col], r["label_date"]), np.nan), axis=1)
        y = pd.Series(y, name="y")

        # Drop rows with missing y just in case
        mask = ~y.isna()
        wide = wide.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

        meta = wide[[entity_col, "label_date"]].copy()

        # Features only
        X = wide.drop(columns=[entity_col, "label_date"])

        X_rows.append(X)
        y_rows.append(y)
        meta_rows.append(meta)

    X_all = pd.concat(X_rows, ignore_index=True) if X_rows else pd.DataFrame()
    y_all = pd.concat(y_rows, ignore_index=True) if y_rows else pd.Series(dtype=float)
    meta_all = pd.concat(meta_rows, ignore_index=True) if meta_rows else pd.DataFrame()

    X_all = X_all.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X_all, y_all.astype(int), meta_all

# Temporal CV driver main execution block
def run_experiment_general(
    data: pd.DataFrame,
    exp_config: dict,
    temporal_spec: dict,
    *,
    entity_col: str,
    date_col: str,
    target_col: str,
    feature_cols: list[str] | None = None,
    drop_cols: list[str] | None = None,
):
    np.random.seed(exp_config["seed"])
    # Logger setup 
    lg.basicConfig(
        filename='logs/test_log.txt', filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=lg.INFO
    )

    def log_warning(message, category, filename, lineno, file=None, line=None):
        lg.warning(f'{filename}:{lineno} - {category.__name__} - {message}')
    warnings.showwarning = log_warning

    today = datetime.now().strftime('%Y%m%d')
    experiment_hash = uuid.uuid4().hex
    log_dir = f'logs/exp_{today}_{experiment_hash[-4:]}'
    os.mkdir(log_dir)

    lg.info(f'Logging experiment {today}_{experiment_hash[-4:]}')
    lg.info(f'Experiment config location: {exp_config["path"]}')
    lg.info(f'Entities: {entity_col} | Date: {date_col} | Target: {target_col}')

    # Ensure datetime
    data = ensure_monthly_datetime(data, date_col)
    
    entity_filter = exp_config.get("entity_filter", None)
    if entity_filter is not None:
        data = data[data[entity_col].isin(entity_filter)].copy()

    # Timecuts 
    h = temporal_spec["h"]
    yval_end_dates = pd.date_range(
        start=temporal_spec["label_end"],
        end=temporal_spec["label_start"] +
            temporal_spec["feature_timeframe_months"] +
            (2 * pd.DateOffset(months=h)) +
            temporal_spec["model_update_freq_months"] -
            pd.DateOffset(months=2),
        freq=-1 * temporal_spec["model_update_freq_months"]
    )

    lg.info(f'Experiment consists of {len(yval_end_dates)} timecuts for h={h}')

    # Initialize model grid
    model_ids, models, hyperparams = [], [], []
    model_dict = {}

    for model_name, params in exp_config["model_grid"].items():
        Model = eval(model_name)
        keys, values = zip(*params.items())
        num_models = prod([len(v) for v in values])

        lg.info(f'Training {num_models} {model_name} models')
        for combination in itertools.product(*values):
            kwarg_dict = dict(zip(keys, combination))
            model = Model(**kwarg_dict)
            model_id = uuid.uuid4().hex[-8:]
            model_ids.append(model_id)
            models.append(model_name)
            hyperparams.append(str(kwarg_dict))
            model_dict[model_id] = model

    model_guide = pd.DataFrame({
        "model_id": model_ids,
        "model": models,
        "hyperparams": hyperparams,
    })
    model_guide.to_csv(f"{log_dir}/model_guide.csv", index=False)

    trained_models = {}
    results_df = pd.DataFrame()

    # Sort by date for index slicing assumptions 
    data = data.sort_values(by=[date_col, entity_col]).reset_index(drop=True)

    # Iterate timecuts
    for m, yval_end in enumerate(yval_end_dates, start=1):
        # Find indices similarly, but robustly (avoid relying on .index[0] if missing)
        if not (data[date_col] == yval_end).any():
            lg.warning(f"Skipping timecut yval_end={yval_end.date()} (date not in data).")
            continue

        # yval start = first date > (yval_end - update_freq)
        cutoff = yval_end - temporal_spec["model_update_freq_months"]
        yval_start = data.loc[data[date_col] > cutoff, date_col].min()

        # training borders
        ytrain_end = yval_start - pd.DateOffset(months=h)
        ytrain_start = max(
            ytrain_end - temporal_spec["max_training_history_years"] + pd.DateOffset(months=1),
            temporal_spec["label_start"] + temporal_spec["feature_timeframe_months"] + pd.DateOffset(months=h-1),
        )

        # Build lists of label dates
        all_dates = sorted(data[date_col].unique())
        ytrain_dates = [d for d in all_dates if (d >= ytrain_start and d <= ytrain_end)]
        yval_dates = [d for d in all_dates if (d >= yval_start and d <= yval_end)]

        if len(ytrain_dates) == 0 or len(yval_dates) == 0:
            lg.warning(f"Skipping timecut yval_end={yval_end.date()} (empty train/val date set).")
            continue

        # Build supervised datasets 
        X_train, y_train, meta_train = build_supervised_matrix(
            data,
            ytrain_dates,
            temporal_spec,
            entity_col=entity_col,
            date_col=date_col,
            target_col=target_col,
            feature_cols=feature_cols,
            drop_cols=drop_cols,
        )

        X_val, y_val, meta_val = build_supervised_matrix(
            data,
            yval_dates,
            temporal_spec,
            entity_col=entity_col,
            date_col=date_col,
            target_col=target_col,
            feature_cols=feature_cols,
            drop_cols=drop_cols,
        )

        # Column alignment
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

        lg.info(
            f"Timecut {m}: yval_end={yval_end.date()} | "
            f"train rows={X_train.shape[0]} | val rows={X_val.shape[0]}"
        )

        # Prepare output frame for this timecut
        timecut_out = meta_val.rename(columns={"label_date": "eval_date"}).copy()
        timecut_out["ytrue"] = y_val.values

        trained_models[yval_end.strftime("%Y%m%d")] = {}

        # Fit/predict for each model
        for model_id in model_guide["model_id"].tolist():
            model = model_dict[model_id]

            try:
                model.fit(X_train, y_train)
                trained_models[yval_end.strftime("%Y%m%d")][model_id] = model

                y_pred = model.predict_proba(X_val)
                class1_i = np.where(model.classes_ == 1)[0][0]
                p1 = y_pred[:, class1_i]

            except Exception as e:
                # Fallback like your code: dummy / mean prediction
                lg.error(f"Model {model_id} failed: {repr(e)}. Using mean(y_train).")
                p1 = np.ones(X_val.shape[0]) * float(np.mean(y_train))

            timecut_out[model_id] = p1

        # Append
        results_df = pd.concat([results_df, timecut_out], ignore_index=True)

        # Save snapshot models per timecut
        with open(f"{log_dir}/trained_models_timecut_{m}_{yval_end.strftime('%Y%m%d')}.pickle", "wb") as f:
            pickle.dump(trained_models[yval_end.strftime("%Y%m%d")], f)

        lg.info(f"Completed timecut yval_end={yval_end.date()}")

    # Save outputs
    results_df.to_csv(f"{log_dir}/results.csv", index=False)
    with open(f"{log_dir}/trained_models_all.pickle", "wb") as f:
        pickle.dump(trained_models, f)

    lg.info("All training & prediction complete.")
    lg.shutdown()

    return log_dir

if __name__ == "__main__":

    config_path = sys.argv[1]

    exp_config, temporal_config = parse_config(config_path)

    data_processed = import_data()
    # Dataset mapping 
    ENTITY = "user"
    DATE = "date"
    TARGET = exp_config["targets"][0]

    # Specify feature columns explicitly
    feature_cols = None

    # Sometimes flat files have this weird thing
    drop_cols = ["Unnamed: 0"]

    temp_keys, temp_values = zip(*temporal_config.items())

    for temp_combo in itertools.product(*temp_values):

        temporal_spec = dict(zip(temp_keys, temp_combo))

        run_experiment_general(
            data_processed,
            exp_config,
            temporal_spec,
            entity_col=ENTITY,
            date_col=DATE,
            target_col=TARGET,
            feature_cols=feature_cols,
            drop_cols=drop_cols
        )
    