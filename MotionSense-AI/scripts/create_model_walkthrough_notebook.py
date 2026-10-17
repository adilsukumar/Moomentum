"""Generate the self-contained MotionSense model walkthrough notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "MotionSense_Model_A_to_Z.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    md(r"""
    # MotionSense: dog-behavior model, from raw X/Y/Z signals to deployment

    **A reproducible learning notebook for the model in this repository**

    This notebook is written as a lab record I can explain in my own words. It does not invent training output: cells either compute a result, load a result already produced by the repository, or label a number as a design choice.

    By the end I should be able to explain:

    1. what the physical accelerometer and gyroscope **x, y, z axes** measure;
    2. what machine-learning **`X` and `y`** mean;
    3. how 100 raw sensor samples become 64 features;
    4. why the production system uses two LightGBM classifiers;
    5. how dog-grouped cross-validation prevents leakage;
    6. how to train, evaluate, save, load, and run the model;
    7. what the reported accuracy does—and does not—prove.

    > The model classifies motion behavior. It does **not** diagnose illness. The dataset has no disease labels.
    """),
    md(r"""
    ## 0. The whole pipeline in one picture

    ```text
    collar IMU at 100 Hz
            │
            ├── accelerometer: ANeck_x, ANeck_y, ANeck_z  (g)
            └── gyroscope:     GNeck_x, GNeck_y, GNeck_z  (degrees/s)
            │
       1-second windows (100 samples), 0.5-second stride, ≥80% label purity
            │
       add accel/gyro magnitude → 8 channels
            │
       8 statistics per channel → X has 64 columns
            │
       coarse LightGBM → Active / Resting / Other
            │
            └── only if Active: fine LightGBM → 7 active behaviors
            │
       confidence gate / smoothing / app summary
    ```

    The deployed artifacts are `export/behavior_coarse_lgbm.joblib` and `export/behavior_active_fine_lgbm.joblib`, with equivalent ONNX files for mobile inference.
    """),
    md(r"""
    ## 1. Colab setup and reproducibility

    This notebook is **Google Colab ready**. The first code cell installs missing packages. The raw CSV is about 2 GB, so Google Drive is the recommended data source; browser upload is included as a fallback. For full training, select a **High-RAM** runtime if Colab offers one (`Runtime → Change runtime type`). A GPU is not required because LightGBM runs well on CPU.

    Runtime choices are intentionally explicit:

    - `drive` — recommended for the large CSV; mount Drive and point to the file;
    - `upload` — choose the CSV from the browser (slow and memory-heavy for ~2 GB);
    - `existing` — use a path already present under `/content` or in a local checkout.

    The notebook contains its own preprocessing implementation, so `scripts/features.py` is not required in Colab.
    """),
    code(r"""
    from pathlib import Path
    import importlib.util, json, os, subprocess, sys, time, warnings

    IN_COLAB = importlib.util.find_spec("google.colab") is not None
    required = {
        "numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib",
        "seaborn": "seaborn", "sklearn": "scikit-learn",
        "lightgbm": "lightgbm", "joblib": "joblib",
    }
    missing = [pip_name for module, pip_name in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from numpy.lib.stride_tricks import sliding_window_view
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import LabelEncoder
    from lightgbm import LGBMClassifier

    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid")
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)

    if IN_COLAB:
        ROOT = Path("/content/motionsense")
        ROOT.mkdir(parents=True, exist_ok=True)
    else:
        candidates = [Path.cwd(), Path.cwd().parent]
        ROOT = next((p for p in candidates if (p / "DogMoveData.csv").exists()), Path.cwd())

    print("Running in Colab:", IN_COLAB)
    print("Workspace:", ROOT)
    print("Python:", sys.version.split()[0])
    print("Seed:", RANDOM_STATE)
    """),
    code(r"""
    # Self-contained copy of the production preprocessing contract.
    SAMPLE_RATE_HZ = 100
    WINDOW = 100
    STRIDE = 50
    MAX_GAP_SEC = 0.015
    PURITY = 0.8

    NECK_AXES = ["ANeck_x", "ANeck_y", "ANeck_z", "GNeck_x", "GNeck_y", "GNeck_z"]
    BACK_AXES = ["ABack_x", "ABack_y", "ABack_z", "GBack_x", "GBack_y", "GBack_z"]
    COARSE_MAP = {
        "Walking": "Active", "Trotting": "Active", "Pacing": "Active",
        "Galloping": "Active", "Playing": "Active", "Jumping": "Active", "Tugging": "Active",
        "Standing": "Resting", "Sitting": "Resting", "Lying chest": "Resting", "Panting": "Resting",
        "Shaking": "Other", "Sniffing": "Other", "Eating": "Other", "Drinking": "Other",
        "Bowing": "Other", "Carrying object": "Other",
    }
    VALID_BEHAVIORS = set(COARSE_MAP)

    def add_magnitude(df, axes, out_name):
        x, y, z = (df[a].to_numpy() for a in axes)
        df[out_name] = np.sqrt(x*x + y*y + z*z)

    def channel_list(use_back=False):
        channels = list(NECK_AXES) + ["ANeck_mag", "GNeck_mag"]
        if use_back:
            channels += list(BACK_AXES) + ["ABack_mag", "GBack_mag"]
        return channels

    def extract_window_features(window_arr):
        mean = window_arr.mean(axis=1)
        std = window_arr.std(axis=1)
        mn = window_arr.min(axis=1)
        mx = window_arr.max(axis=1)
        rms = np.sqrt((window_arr ** 2).mean(axis=1))
        signs = np.sign(window_arr)
        zcr = (signs[:, :-1, :] * signs[:, 1:, :] < 0).sum(axis=1).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(window_arr, axis=1))
        domfreq = spectrum.argmax(axis=1).astype(np.float32)
        dompow = spectrum.max(axis=1)
        return np.concatenate([mean, std, mn, mx, rms, zcr, domfreq, dompow], axis=1).astype(np.float32)

    def feature_names(channels):
        stats = ["mean", "std", "min", "max", "rms", "zcr", "domfreq", "dompow"]
        return [f"{channel}_{stat}" for stat in stats for channel in channels]

    def build_windows(df, channels):
        raw_windows, fine, coarse, dogs, tests, centers = [], [], [], [], [], []
        for (dog, test), g in df.groupby(["DogID", "TestNum"], sort=False):
            g = g.sort_values("t_sec")
            t = g["t_sec"].to_numpy()
            if len(t) < WINDOW:
                continue
            run_ids = (np.diff(t, prepend=t[0] - 0.01) > MAX_GAP_SEC).cumsum()
            values = g[channels].to_numpy(dtype=np.float32)
            labels = g["Behavior_1"].to_numpy()
            coarse_labels = g["Coarse"].to_numpy()
            for run_id in np.unique(run_ids):
                mask = run_ids == run_id
                if mask.sum() < WINDOW:
                    continue
                seg, seg_t = values[mask], t[mask]
                seg_y, seg_c = labels[mask], coarse_labels[mask]
                windows = sliding_window_view(seg, WINDOW, axis=0)[::STRIDE].transpose(0, 2, 1)
                y_windows = sliding_window_view(seg_y, WINDOW)[::STRIDE]
                c_windows = sliding_window_view(seg_c, WINDOW)[::STRIDE]
                t_windows = sliding_window_view(seg_t, WINDOW)[::STRIDE]
                for i in range(len(windows)):
                    vals, counts = np.unique(y_windows[i], return_counts=True)
                    winner = counts.argmax()
                    if counts[winner] / WINDOW < PURITY:
                        continue
                    cvals, ccounts = np.unique(c_windows[i], return_counts=True)
                    raw_windows.append(windows[i]); fine.append(vals[winner])
                    coarse.append(cvals[ccounts.argmax()]); dogs.append(dog); tests.append(test)
                    centers.append(t_windows[i][WINDOW // 2])
        if not raw_windows:
            return None
        return {"windows": np.stack(raw_windows), "fine": np.array(fine),
                "coarse": np.array(coarse), "dog_id": np.array(dogs),
                "test_num": np.array(tests), "t_center": np.array(centers)}

    print("Preprocessing contract loaded; expected feature count:",
          len(feature_names(channel_list(False))))
    """),
    md(r"""
    ## 2. What x, y, and z mean

    There are two different uses of these letters.

    ### Physical x/y/z axes

    The collar contains an IMU (inertial measurement unit):

    - `ANeck_x/y/z` are linear acceleration components, stored in units compatible with **g**. At rest, gravity still contributes about 1 g across the three axes depending on collar orientation.
    - `GNeck_x/y/z` are angular-velocity components in **degrees per second**. They measure rotation around each axis.
    - x, y, and z are coordinates fixed to the sensor, not universal “forward/side/up” directions. The interpretation depends on how the sensor was mounted. Rotation or collar slippage changes individual axes.
    - Magnitude, `sqrt(x² + y² + z²)`, is less sensitive to orientation. That is why the pipeline adds acceleration and gyroscope magnitude channels.

    Examples: trotting tends to create periodic acceleration; shaking creates fast rotational/acceleration changes; resting has lower variation. Exact signatures overlap, so the classifier uses all channels and many statistics together.

    ### Machine-learning X and y

    - **`X`** (capital X) is the feature matrix. One row is one accepted one-second window; its 64 columns summarize motion.
    - **`y`** (lowercase y) is the target label the model learns to predict, such as `Active`, `Resting`, or `Other`.
    - **groups** is `DogID`, used only to keep each validation dog out of that fold's training data.

    Shapes: `X.shape == (number_of_windows, 64)` and `y.shape == (number_of_windows,)`.
    """),
    md(r"""
    ## 3. Connect or upload the dataset

    For a ~2 GB dataset, Drive is much more reliable than browser upload. Edit only the configuration values below and run the cell. When using `upload`, Colab holds the upload in memory while receiving it, so a High-RAM runtime may be necessary.
    """),
    code(r"""
    # Default: use the dataset already stored on this Windows computer.
    # This works in local Jupyter, VS Code, or a Colab notebook connected to a local runtime.
    DATA_SOURCE = "existing"  # Alternatives: "drive" or "upload"
    DATASET_PATH = "DogMoveData.csv"

    if DATA_SOURCE == "drive":
        if not IN_COLAB:
            raise RuntimeError("The drive option is available only in Google Colab.")
        from google.colab import drive
        drive.mount("/content/drive")
        CSV_PATH = Path(DATASET_PATH)
    elif DATA_SOURCE == "upload":
        if not IN_COLAB:
            raise RuntimeError("Use DATA_SOURCE='existing' outside Colab.")
        from google.colab import files
        print("Select DogMoveData.csv. Large browser uploads may take a long time.")
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError("No file uploaded.")
        uploaded_name = next(iter(uploaded))
        CSV_PATH = Path("/content") / uploaded_name
    elif DATA_SOURCE == "existing":
        CSV_PATH = Path(DATASET_PATH).expanduser()
    else:
        raise ValueError("DATA_SOURCE must be 'drive', 'upload', or 'existing'.")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}. Edit DATASET_PATH or choose upload.")
    print(f"Dataset ready: {CSV_PATH} ({CSV_PATH.stat().st_size / 1024**3:.2f} GiB)")
    """),
    md(r"""
    ## 4. Inspect the raw data safely

    The CSV has neck and back IMUs, annotations, dog identity, test number, and timestamps. The product model deliberately uses only neck channels because a collar is more practical than requiring both a collar and a back harness.
    """),
    code(r"""
    raw_preview = pd.read_csv(CSV_PATH, nrows=20_000)
    print(f"CSV size: {CSV_PATH.stat().st_size / 1024**3:.2f} GiB")
    print("Preview shape:", raw_preview.shape)
    display(raw_preview.head())
    display(raw_preview.dtypes.to_frame("dtype"))
    """),
    code(r"""
    sensor_cols = NECK_AXES
    display(raw_preview[sensor_cols].describe().T)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    plot_n = min(1500, len(raw_preview))
    raw_preview.iloc[:plot_n].plot(x="t_sec", y=NECK_AXES[:3], ax=axes[0], lw=0.8)
    raw_preview.iloc[:plot_n].plot(x="t_sec", y=NECK_AXES[3:], ax=axes[1], lw=0.8)
    axes[0].set_ylabel("Acceleration (g)")
    axes[1].set_ylabel("Angular velocity (deg/s)")
    axes[1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    """),
    md(r"""
    ## 5. Labels: what the model predicts

    `Behavior_1` contains the primary annotated behavior. Annotation artifacts such as `Synchronization`, `Extra_Synchronization`, `<undefined>`, and missing values are excluded.

    The 17 valid fine labels are mapped to three owner-friendly motion categories:

    - **Active:** Walking, Trotting, Pacing, Galloping, Playing, Jumping, Tugging
    - **Resting:** Standing, Sitting, Lying chest, Panting
    - **Other:** Shaking, Sniffing, Eating, Drinking, Bowing, Carrying object

    Panting is placed under Resting because a neck IMU sees it mostly as a still-body state and cannot reliably separate it from sitting/standing/lying. This is a sensor-observability decision, not a medical claim.
    """),
    code(r"""
    label_map = pd.DataFrame(sorted(COARSE_MAP.items()), columns=["fine_behavior", "coarse_class"])
    display(label_map)
    display(raw_preview["Behavior_1"].fillna("<missing>").value_counts().head(20).to_frame("rows_in_preview"))
    """),
    md(r"""
    ## 6. Cleaning and windowing

    Raw rows are not treated as independent examples. The model uses windows because behavior is a pattern over time.

    Rules in `scripts/features.py`:

    - sample rate: 100 Hz;
    - window: 100 samples = 1 second;
    - stride: 50 samples = 0.5 seconds, so adjacent windows overlap 50%;
    - never cross a `DogID` or `TestNum` boundary;
    - split a run when the timestamp gap exceeds 0.015 seconds;
    - keep a window only when at least 80% of its samples share the same fine label.

    Overlap increases training examples but makes neighboring windows correlated. That is another reason a random row/window split would be misleading.
    """),
    code(r"""
    def clean_frame(df):
        df = df[df["Behavior_1"].isin(VALID_BEHAVIORS)].copy()
        df["Coarse"] = df["Behavior_1"].map(COARSE_MAP)
        add_magnitude(df, ["ANeck_x", "ANeck_y", "ANeck_z"], "ANeck_mag")
        add_magnitude(df, ["GNeck_x", "GNeck_y", "GNeck_z"], "GNeck_mag")
        return df

    # Read until the preview includes usable labeled motion; increase if necessary.
    usecols = ["DogID", "TestNum", "t_sec", "Behavior_1"] + NECK_AXES
    demo_raw = pd.read_csv(CSV_PATH, usecols=usecols, nrows=750_000)
    demo_df = clean_frame(demo_raw)
    print("Raw demo rows:", len(demo_raw))
    print("Valid labeled rows:", len(demo_df))
    display(demo_df.head())
    """),
    code(r"""
    neck_channels = channel_list(use_back=False)
    demo_windows = build_windows(demo_df, neck_channels)
    if demo_windows is None:
        raise RuntimeError("No complete labeled window found; increase nrows in the previous cell.")

    print("Window tensor:", demo_windows["windows"].shape)
    print("Fine labels:", demo_windows["fine"].shape)
    print("First window dog/test/time/labels:",
          demo_windows["dog_id"][0], demo_windows["test_num"][0],
          demo_windows["t_center"][0], demo_windows["fine"][0],
          demo_windows["coarse"][0])
    """),
    md(r"""
    ## 7. Feature engineering: 100 × 8 becomes 64

    Eight channels are used: six raw neck axes plus acceleration magnitude and gyroscope magnitude. Eight features are calculated for every channel:

    1. **mean** — average level/orientation;
    2. **standard deviation** — variability;
    3. **minimum** — negative extreme;
    4. **maximum** — positive extreme;
    5. **RMS** — signal energy, `sqrt(mean(x²))`;
    6. **zero-crossing count** — number of sign changes, a rough activity/frequency cue;
    7. **dominant FFT bin** — strongest frequency bin (with a 1-second, 100 Hz window, bins are spaced 1 Hz apart; the implementation stores the bin index, numerically equal to Hz here);
    8. **dominant FFT magnitude** — strength of that dominant component.

    Therefore `8 channels × 8 summaries = 64 features`. This compact representation is much cheaper on a phone than feeding all raw samples to a deep network.

    Important nuance: the FFT includes the DC bin, so a large mean/gravity component can become the dominant frequency. This is a known modeling choice, not a universally optimal one.
    """),
    code(r"""
    X_demo = extract_window_features(demo_windows["windows"])
    y_demo = demo_windows["coarse"]
    groups_demo = demo_windows["dog_id"]
    names = feature_names(neck_channels)

    X_demo_df = pd.DataFrame(X_demo, columns=names)
    print("X shape:", X_demo_df.shape)
    print("y shape:", y_demo.shape)
    print("groups shape:", groups_demo.shape)
    display(X_demo_df.head())
    """),
    code(r"""
    i = 0
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    t = np.arange(WINDOW) / SAMPLE_RATE_HZ
    axes[0].plot(t, demo_windows["windows"][i, :, :3])
    axes[0].legend(neck_channels[:3]); axes[0].set_ylabel("g")
    axes[1].plot(t, demo_windows["windows"][i, :, 3:6])
    axes[1].legend(neck_channels[3:6]); axes[1].set_ylabel("deg/s")
    axes[1].set_xlabel("Seconds within window")
    fig.suptitle(f"One training example: {demo_windows['fine'][i]} → {demo_windows['coarse'][i]}")
    plt.tight_layout()
    """),
    md(r"""
    ## 8. Why LightGBM?

    LightGBM is gradient-boosted decision trees. A single tree asks threshold questions such as “is gyroscope variability above this value?” A boosted ensemble builds trees sequentially so later trees correct earlier errors. Class probabilities come from the accumulated scores.

    Production coarse-model settings:

    - `n_estimators=300`: 300 boosting rounds;
    - `num_leaves=31`: upper bound on leaf complexity per tree;
    - `class_weight='balanced'`: rare classes receive more weight;
    - `random_state=42`: reproducibility;
    - `n_jobs=-1`: use available CPU cores;
    - `verbosity=-1`: quieter output.

    The active-only model uses `num_leaves=63` because seven fine classes need more capacity. Tree models suit engineered tabular features, train quickly, expose feature importance, and export cleanly to ONNX.
    """),
    md(r"""
    ## 9. Optional: upload the saved model and result artifacts

    Dataset exploration and retraining need only the CSV. To inspect the exact pretrained models and saved metrics from this repository, upload any of these files when prompted:

    - `behavior_coarse_lgbm.joblib`
    - `behavior_active_fine_lgbm.joblib`
    - `lightgbm_results.json`
    - `export_summary.json`
    - `feature_names.json`

    Set `UPLOAD_ARTIFACTS=True` in Colab. You may select all five files in one upload dialog. If the notebook is running inside the original repository, it finds them automatically.
    """),
    code(r"""
    ARTIFACT_DIR = ROOT / "uploaded_artifacts"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_ARTIFACTS = False

    if UPLOAD_ARTIFACTS:
        if not IN_COLAB:
            raise RuntimeError("Place artifacts in the repository or uploaded_artifacts outside Colab.")
        from google.colab import files
        uploaded_artifacts = files.upload()
        for filename, payload in uploaded_artifacts.items():
            (ARTIFACT_DIR / Path(filename).name).write_bytes(payload)

    def find_artifact(filename, repository_subdir=None):
        candidates = [ARTIFACT_DIR / filename, ROOT / filename]
        if repository_subdir:
            candidates.insert(0, ROOT / repository_subdir / filename)
        return next((path for path in candidates if path.exists()), None)

    wanted = ["behavior_coarse_lgbm.joblib", "behavior_active_fine_lgbm.joblib",
              "lightgbm_results.json", "export_summary.json", "feature_names.json"]
    artifact_status = {name: find_artifact(name, "export" if name != "lightgbm_results.json" else "models")
                       for name in wanted}
    display(pd.Series({name: str(path) if path else "not supplied"
                       for name, path in artifact_status.items()}, name="location").to_frame())
    """),
    md(r"""
    ## 10. Validation without dog leakage

    A random train/test split is wrong here: overlapping windows from the same dog and session can land in both sets, letting the model memorize that dog's collar orientation and gait.

    `GroupKFold(n_splits=5)` assigns whole dogs to folds. In each fold, no `DogID` appears in both training and validation. This is often described as dog-grouped or leave-dogs-out validation. It is not literal one-dog-at-a-time LODO because each of the five folds holds out a group of dogs.

    Accuracy measures total correctness. **Macro F1** calculates F1 per class and averages equally, so minority-class failure cannot hide behind a common class.
    """),
    code(r"""
    metrics_path = artifact_status["lightgbm_results.json"]
    if metrics_path:
        metrics = json.loads(metrics_path.read_text())
        summary = pd.DataFrame({
            name: {"accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}
            for name, result in metrics.items()
        }).T
        display(summary.style.format("{:.3%}"))
        print("These are saved grouped-CV results loaded from:", metrics_path)
    else:
        print("No metrics JSON supplied. Upload it above or run grouped CV later in the notebook.")
    """),
    md(r"""
    ### Interpreting the saved results

    The repository reports the neck-only LightGBM coarse model at roughly **95.6% accuracy** and **94.9% macro F1** over about 119,826 accepted windows. The fine 17-class problem is much harder (roughly 67% accuracy in the LightGBM comparison), chiefly because rare actions have very little support and several stationary behaviors overlap strongly in IMU space.

    These results estimate generalization to held-out dogs from the same study—not guaranteed performance for every breed, collar fit, home environment, sensor calibration, or rapid behavior transition.
    """),
    code(r"""
    # Demonstrate the invariant each grouped split must satisfy.
    unique_demo_dogs = np.unique(groups_demo)
    if len(unique_demo_dogs) >= 5:
        splitter = GroupKFold(n_splits=5)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_demo, y_demo, groups_demo), 1):
            train_dogs = set(groups_demo[train_idx])
            valid_dogs = set(groups_demo[valid_idx])
            assert train_dogs.isdisjoint(valid_dogs)
            print(f"fold {fold}: {len(train_dogs)} train dogs, {len(valid_dogs)} validation dogs")
    else:
        print("The bounded preview has fewer than 5 dogs; the full pipeline uses all 45 dogs.")
    """),
    md(r"""
    ## 11. Inspect the model that was actually exported

    A joblib artifact is a dictionary containing the fitted LightGBM estimator and its `LabelEncoder`. Never load an untrusted pickle/joblib file: deserialization can execute code.
    """),
    code(r"""
    coarse_path = artifact_status["behavior_coarse_lgbm.joblib"]
    fine_path = artifact_status["behavior_active_fine_lgbm.joblib"]
    HAVE_PRETRAINED_MODELS = bool(coarse_path and fine_path)

    if HAVE_PRETRAINED_MODELS:
        coarse_bundle = joblib.load(coarse_path)
        fine_bundle = joblib.load(fine_path)
        coarse_model = coarse_bundle["model"]
        coarse_encoder = coarse_bundle["label_encoder"]
        active_model = fine_bundle["model"]
        active_encoder = fine_bundle["label_encoder"]
        print("Coarse classes:", coarse_encoder.classes_)
        print("Active fine classes:", active_encoder.classes_)
        print("Input features:", coarse_model.n_features_in_)
        print("Coarse boosting rounds:", coarse_model.booster_.current_iteration())
        display(pd.Series(coarse_model.get_params(), name="value").to_frame())
    else:
        coarse_model = active_model = coarse_encoder = active_encoder = None
        print("Pretrained joblib files were not supplied. Model inspection and demo inference will be skipped.")
    """),
    code(r"""
    if HAVE_PRETRAINED_MODELS:
        importance = pd.Series(
            coarse_model.booster_.feature_importance(importance_type="gain"),
            index=names,
            name="gain",
        ).sort_values(ascending=False)
        importance = 100 * importance / importance.sum()
        display(importance.head(20).to_frame("percent_of_total_gain"))
        importance.head(20).sort_values().plot.barh(figsize=(10, 7))
        plt.xlabel("Percent of total split gain")
        plt.title("Coarse model feature importance")
        plt.tight_layout()
    else:
        print("Skipped: upload both pretrained joblib files in section 9.")
    """),
    md(r"""
    Feature importance describes how much training loss was reduced by splits using each feature. It is not causality, and correlated features can share or steal importance from one another. Permutation importance or SHAP can provide richer explanations, but must be computed on held-out dogs.
    """),
    md(r"""
    ## 12. Run the hierarchical predictor

    The coarse model always runs first. The fine model is meaningful only inside `Active`. A confidence threshold can abstain rather than presenting a weak guess. The threshold below is a demonstration choice; production calibration should use out-of-fold predictions.
    """),
    code(r"""
    def predict_hierarchical(X, confidence_threshold=0.60):
        X = np.asarray(X, dtype=np.float32)
        coarse_p = coarse_model.predict_proba(X)
        coarse_ids = coarse_p.argmax(axis=1)
        coarse_labels = coarse_encoder.inverse_transform(coarse_ids)

        rows = []
        for i, coarse_label in enumerate(coarse_labels):
            coarse_conf = float(coarse_p[i].max())
            result = {
                "coarse_label": coarse_label,
                "coarse_confidence": coarse_conf,
                "fine_label": None,
                "fine_confidence": None,
                "display_label": "Uncertain" if coarse_conf < confidence_threshold else coarse_label,
            }
            if coarse_label == "Active" and coarse_conf >= confidence_threshold:
                fine_p = active_model.predict_proba(X[i:i+1])[0]
                fine_id = int(fine_p.argmax())
                result["fine_label"] = active_encoder.inverse_transform([fine_id])[0]
                result["fine_confidence"] = float(fine_p[fine_id])
                if result["fine_confidence"] >= confidence_threshold:
                    result["display_label"] = result["fine_label"]
            rows.append(result)
        return pd.DataFrame(rows)

    if HAVE_PRETRAINED_MODELS:
        demo_predictions = predict_hierarchical(X_demo[:20])
        demo_predictions.insert(0, "true_fine", demo_windows["fine"][:20])
        demo_predictions.insert(1, "true_coarse", demo_windows["coarse"][:20])
        display(demo_predictions)
    else:
        print("Skipped: upload pretrained models, or complete full training below and rerun this section.")
    """),
    md(r"""
    ## 13. Full training—explicit opt-in

    This cell reconstructs the production-style coarse model from the complete CSV and saves a new artifact with a different filename. It can be memory-intensive because the current pipeline stacks all accepted raw windows before feature extraction.

    Training on all data is for the final artifact. Honest performance comes from grouped cross-validation, not evaluating this final model on the same data it saw.
    """),
    code(r"""
    RUN_FULL_TRAINING = False  # Change only when ready for a long, memory-heavy run.

    if RUN_FULL_TRAINING:
        all_sensor_cols = NECK_AXES
        dtypes = {c: "float32" for c in all_sensor_cols}
        dtypes.update({"DogID": "int16", "TestNum": "int16", "t_sec": "float32"})
        full_df = pd.read_csv(
            CSV_PATH,
            usecols=["DogID", "TestNum", "t_sec", "Behavior_1"] + all_sensor_cols,
            dtype=dtypes,
        )
        full_df = clean_frame(full_df)
        full_windows = build_windows(full_df, neck_channels)
        X = extract_window_features(full_windows["windows"])
        y = full_windows["coarse"]
        groups = full_windows["dog_id"]

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        coarse_retrained = LGBMClassifier(
            n_estimators=300, num_leaves=31, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
        )
        started = time.time()
        coarse_retrained.fit(X, y_encoded)

        active_mask = y == "Active"
        active_encoder_retrained = LabelEncoder()
        active_y = active_encoder_retrained.fit_transform(full_windows["fine"][active_mask])
        active_retrained = LGBMClassifier(
            n_estimators=300, num_leaves=63, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
        )
        active_retrained.fit(X[active_mask], active_y)

        output_dir = ROOT / "my_trained_models"
        output_dir.mkdir(parents=True, exist_ok=True)
        coarse_out = output_dir / "my_retrained_coarse_lgbm.joblib"
        active_out = output_dir / "my_retrained_active_fine_lgbm.joblib"
        joblib.dump({"model": coarse_retrained, "label_encoder": encoder,
                     "feature_names": names}, coarse_out)
        joblib.dump({"model": active_retrained, "label_encoder": active_encoder_retrained,
                     "feature_names": names}, active_out)
        # Make the freshly trained models available to the earlier inference function.
        coarse_model, coarse_encoder = coarse_retrained, encoder
        active_model, active_encoder = active_retrained, active_encoder_retrained
        HAVE_PRETRAINED_MODELS = True
        print(f"Saved both models under {output_dir} after {(time.time()-started)/60:.1f} minutes")

        if IN_COLAB:
            from google.colab import files
            print("Downloading trained artifacts to your computer...")
            files.download(str(coarse_out))
            files.download(str(active_out))
    else:
        print("Skipped. Set RUN_FULL_TRAINING=True to rebuild the artifact.")
    """),
    md(r"""
    ## 14. Grouped cross-validation training template

    Use this logic when I need to regenerate performance numbers. It trains a fresh model in every fold and concatenates predictions so each prediction comes from a model that never saw that dog.
    """),
    code(r"""
    def grouped_cv_predictions(X, y, dog_ids, n_splits=5):
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        oof_pred = np.full(len(y_encoded), -1, dtype=int)
        splitter = GroupKFold(n_splits=n_splits)

        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y_encoded, dog_ids), 1):
            assert set(dog_ids[train_idx]).isdisjoint(set(dog_ids[valid_idx]))
            fold_model = LGBMClassifier(
                n_estimators=300, num_leaves=31, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
            )
            fold_model.fit(X[train_idx], y_encoded[train_idx])
            oof_pred[valid_idx] = fold_model.predict(X[valid_idx]).astype(int)
            print(f"Completed fold {fold}/{n_splits}")

        assert (oof_pred >= 0).all()
        return encoder, y_encoded, oof_pred

    # Example after full feature construction:
    # encoder, y_true, y_pred = grouped_cv_predictions(X, y, groups)
    # print(classification_report(encoder.inverse_transform(y_true),
    #                             encoder.inverse_transform(y_pred), zero_division=0))
    """),
    md(r"""
    ## 15. ONNX deployment and parity

    ONNX is the portable form intended for on-device inference. `export/export_summary.json` records a comparison between native LightGBM and ONNX Runtime on 5,000 samples. Label agreement was 100% for both exported classifiers; small probability differences arise from converter/runtime numeric details.

    The most important deployment contract is not merely “64 floats.” The phone must reproduce the exact channel order, windowing, statistics, FFT behavior, units, and feature order used in training. A correct model with incorrect preprocessing gives incorrect predictions.
    """),
    code(r"""
    export_summary_path = artifact_status["export_summary.json"]
    feature_names_path = artifact_status["feature_names.json"]
    if export_summary_path:
        export_summary = json.loads(export_summary_path.read_text())
        display(pd.DataFrame(export_summary).T)
    else:
        print("No export_summary.json supplied; parity summary skipped.")

    if feature_names_path:
        saved_feature_names = json.loads(feature_names_path.read_text())
        assert saved_feature_names == names
        print("Feature-order contract verified for", len(names), "features.")
    else:
        print("No feature_names.json supplied; generated feature order still has", len(names), "features.")
    """),
    md(r"""
    ## 16. What can go wrong?

    1. **Dog leakage:** random splits inflate results because the same dog's correlated windows appear on both sides.
    2. **Session leakage:** overlapping windows make leakage even easier.
    3. **Class imbalance:** accuracy can hide failure on rare Bowing, Jumping, Galloping, or Tugging examples.
    4. **Domain shift:** lab data is cleaner than uncontrolled home use.
    5. **Mounting/orientation shift:** collar rotation changes x/y/z distributions.
    6. **Hardware mismatch:** sample rate, accelerometer range, gyro range, scaling, clipping, filtering, and units must match.
    7. **Transition bias:** the ≥80% purity rule removes many ambiguous transition windows; reported accuracy mainly describes stable behavior windows.
    8. **Confidence is not automatically calibrated:** a 0.9 tree-ensemble probability is not guaranteed to mean 90% empirical correctness.
    9. **Fine-label scarcity:** no algorithm can reliably learn a class represented by only a handful of held-out examples.
    10. **No disease supervision:** anomaly scores rank unusual motion; they cannot identify a disease.
    """),
    md(r"""
    ## 17. My explanation in 90 seconds

    > I used a 100 Hz neck-mounted accelerometer and gyroscope. I divided each dog's continuous recording into one-second windows with 50% overlap, without crossing test or timestamp gaps, and kept windows whose primary annotation was at least 80% pure. For each window I added orientation-robust acceleration and gyroscope magnitudes, giving eight channels. I summarized each channel with mean, standard deviation, minimum, maximum, RMS, zero crossings, dominant FFT frequency bin, and dominant FFT magnitude, so each example had 64 features. I trained a class-balanced LightGBM classifier for Active, Resting, and Other, then an active-only classifier for seven detailed active behaviors. I evaluated with five-fold GroupKFold by DogID so a validation dog's data never appeared in training. The saved coarse LightGBM result is about 95.6% accuracy and 94.9% macro F1, but that is a lab-dataset estimate and transition windows are filtered. I exported the full-data models to ONNX and verified identical labels on a 5,000-window parity sample. The system classifies movement; it does not diagnose disease.
    """),
    md(r"""
    ## 18. Questions I should be able to answer

    1. Why is gravity visible in accelerometer data even when the dog is still?
    2. Why add vector magnitudes if x/y/z already exist?
    3. Why do we use windows instead of classifying individual samples?
    4. What do `WINDOW=100`, `STRIDE=50`, and `PURITY=0.8` do?
    5. Why are there exactly 64 model inputs?
    6. What is the difference between physical x/y/z and ML `X`/`y`?
    7. Why is GroupKFold by dog more honest than a random split?
    8. Why report macro F1 alongside accuracy?
    9. Why is Panting grouped with Resting?
    10. Why is the 17-class task much harder than the 3-class task?
    11. What does `class_weight='balanced'` change?
    12. Why run the fine model only after an Active prediction?
    13. Why can ONNX probabilities differ slightly while labels still agree?
    14. What preprocessing must the mobile implementation match exactly?
    15. What additional data would be needed before claiming real-world robustness?
    """),
    md(r"""
    ## 19. Final checklist before I present this as my work

    - I ran the cells I intend to discuss and can explain every output.
    - I can derive `8 channels × 8 statistics = 64 features` without notes.
    - I can explain why held-out dogs matter.
    - I say “loaded saved result” when I did not personally rerun full cross-validation.
    - I do not call an in-sample score a validation score.
    - I state that lab performance is not the same as home performance.
    - I never describe the classifier or anomaly layer as a medical diagnostic.

    The fastest way to genuinely own this project is to change one documented choice—such as the window length—rerun grouped validation, predict the trade-off first, and then explain why the result changed.
    """),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {
            "name": "MotionSense_Model_A_to_Z.ipynb",
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
