from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "source"
MPLCONFIG_DIR = BASE_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

MODEL_PATH = SOURCE_DIR / "hybrid_stacking_model.joblib"
METRICS_PATH = SOURCE_DIR / "metrics.json"
META_PATH = SOURCE_DIR / "meta_coefficients.json"
COMPARISON_PATH = SOURCE_DIR / "model_comparison_metrics.csv"


PAGE_OPTIONS = [
    "Home",
    "One-Slide Summary",
    "Project Walkthrough",
    "Report Alignment",
    "Training Pipeline Flowchart",
    "Results Dashboard",
    "Hybrid Model Explanation",
    "Prediction Demo",
    "About / Viva",
]


FLOW_STEPS = [
    {
        "icon": "📦",
        "title": "Raw Kaggle Dataset",
        "subtitle": "Walmart Recruiting - Store Sales Forecasting",
        "what": "The project starts with the Kaggle Walmart sales dataset containing weekly sales, store metadata, and external features.",
        "why": "Retail forecasting depends on historical sales plus contextual variables such as holidays, store type, fuel price, CPI, and unemployment.",
        "report": "This matches the report's proposed retail sales forecasting dataset foundation.",
    },
    {
        "icon": "🔗",
        "title": "Data Merging",
        "subtitle": "train.csv + stores.csv + features.csv",
        "what": "The sales table is merged with store information and external feature data using Store and Date keys.",
        "why": "One modeling table is easier to clean, explore, engineer, split, and pass into machine learning models.",
        "report": "The report described forecasting with combined retail and external features, so this prepares the same modeling view.",
    },
    {
        "icon": "📊",
        "title": "EDA",
        "subtitle": "sales trend, missing values, store/dept analysis",
        "what": "The notebook inspects date coverage, unique stores, departments, missing values, and sales patterns over time.",
        "why": "EDA shows seasonality, holiday spikes, data quality issues, and model-relevant patterns before training.",
        "report": "This supports the report's methodology by validating that the dataset has forecasting structure.",
    },
    {
        "icon": "🧩",
        "title": "Feature Engineering",
        "subtitle": "calendar, holiday, Indian festival, monsoon",
        "what": "Calendar features, holiday flags, Indian festival indicators, and monsoon flags are created from the date column.",
        "why": "Retail demand changes with seasonal, cultural, and weather-like context. These features expose those signals to the models.",
        "report": "The report focused on improving forecast accuracy with meaningful predictors. These features extend that idea.",
    },
    {
        "icon": "⏱️",
        "title": "Lag and Rolling Features",
        "subtitle": "Sales_Lag_1, Sales_Lag_4, Sales_Lag_12, Sales_RollMean_4, Sales_RollStd_12",
        "what": "Past weekly sales values and rolling statistics are created per Store and Dept using only earlier observations.",
        "why": "Lag features tell the model recent demand level, while rolling features summarize short-term trend and volatility.",
        "report": "This strengthens the report's forecasting approach by adding leakage-safe time-series memory.",
    },
    {
        "icon": "🗓️",
        "title": "Time-Based Split",
        "subtitle": "train before 2012, validate on 2012",
        "what": "Training uses records before 2012, and validation uses 2012 records.",
        "why": "Forecasting should learn from the past and predict the future, so random splitting is avoided.",
        "report": "This makes the evaluation more realistic for the report's retail forecasting goal.",
    },
    {
        "icon": "🌳",
        "title": "Base Model Training",
        "subtitle": "CatBoost + LightGBM",
        "what": "CatBoost and LightGBM are trained as gradient boosting base models.",
        "why": "Both are strong tabular forecasting models. CatBoost handles categorical features well, while LightGBM is fast and accurate.",
        "report": "The report proposed gradient boosting models including LightGBM and CatBoost, so this section directly aligns.",
    },
    {
        "icon": "⚖️",
        "title": "MAE-Weighted Ensemble",
        "subtitle": "fixed error-based weighting",
        "what": "The base models are combined using inverse-MAE weights, so lower-error models receive higher influence.",
        "why": "This provides the report-style ensemble baseline and gives a simple comparison against the learned hybrid model.",
        "report": "This exactly implements the report's MAE-weighted voting idea.",
    },
    {
        "icon": "🧠",
        "title": "Stacking Hybrid Model",
        "subtitle": "Ridge meta-learner trained on base model predictions",
        "what": "CatBoost and LightGBM predictions become inputs to a Ridge Regression meta-learner.",
        "why": "Instead of using only fixed weights, stacking lets the final model learn how to combine base predictions from data.",
        "report": "This is an extension of the report, not a mismatch. It upgrades the same ensemble idea into a trained hybrid model.",
    },
    {
        "icon": "📏",
        "title": "Evaluation",
        "subtitle": "MAE, MSE, RMSE, R2",
        "what": "Each model is evaluated using MAE, MSE, RMSE, and R2 on the validation period.",
        "why": "These metrics show error size, squared-error penalty, interpretable RMSE, and overall goodness of fit.",
        "report": "The same metric family appears in the implementation, so results can be defended against the report.",
    },
    {
        "icon": "🚀",
        "title": "Deployment",
        "subtitle": "saved joblib model loaded into Streamlit UI",
        "what": "The trained hybrid artifact is saved as joblib and loaded by the Streamlit prediction interface.",
        "why": "A UI turns the notebook work into a presentable application where users can explain and test the model.",
        "report": "Deployment was not the report's main focus, so the Streamlit dashboard adds project value.",
    },
]


FLOW_STEP_DETAILS = {
    "Raw Kaggle Dataset": [
        (
            "Why this stage is used",
            [
                "It gives the project a real retail forecasting problem instead of a manually created sample dataset.",
                "Weekly sales are available across many stores and departments, so the model learns repeated demand patterns.",
                "External context such as holidays, store profile, fuel price, CPI, and unemployment helps explain sales movement.",
                "The dataset is well known, so reviewers can understand the source and judge the methodology more easily.",
            ],
        ),
        (
            "How it is used",
            [
                "train.csv provides historical Weekly_Sales, Store, Dept, Date, and holiday information.",
                "stores.csv contributes store-level attributes such as Type and Size.",
                "features.csv contributes date-level economic, markdown, temperature, fuel, CPI, unemployment, and holiday features.",
                "The notebook loads these files, converts Date into a usable time column, and prepares them for merging.",
            ],
        ),
        (
            "Features supported",
            [
                "Store and Dept identify where the demand is being forecasted.",
                "Date supports month, week, year, holiday, and seasonality features.",
                "Type and Size describe the store profile.",
                "Economic and external fields can explain demand changes that are not visible from sales history alone.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "The dataset is suitable because sales forecasting needs both historical demand and business context.",
                "The target variable is Weekly_Sales, so the model is a regression model, not a classification model.",
                "The data covers multiple stores and departments, making the project stronger than a single-store forecast.",
            ],
        ),
    ],
    "Data Merging": [
        (
            "Why this stage is used",
            [
                "The raw data is split across separate files, but models need one clean training table.",
                "Merging prevents store metadata and external features from being handled separately during training.",
                "It creates a single row per Store, Dept, and Date with the target and all predictors together.",
                "It reduces confusion during feature engineering because every feature is available in one dataframe.",
            ],
        ),
        (
            "How it is used",
            [
                "train.csv is joined with stores.csv using Store.",
                "The merged sales-store table is joined with features.csv using Store and Date.",
                "Holiday columns from duplicated sources are cleaned into one IsHoliday field.",
                "The final merged table becomes the base dataframe for EDA, feature engineering, and model training.",
            ],
        ),
        (
            "Features supported",
            [
                "Sales columns: Weekly_Sales, Store, Dept, Date.",
                "Store columns: Type and Size.",
                "External columns: Temperature, Fuel_Price, MarkDown fields, CPI, Unemployment, IsHoliday.",
                "Merged keys: Store and Date keep the relationships aligned correctly.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "The merge step converts three disconnected files into one model-ready dataset.",
                "Store joins add business identity; Date joins add time and external context.",
                "A wrong merge would damage the entire model, so this is an important data preparation step.",
            ],
        ),
    ],
    "EDA": [
        (
            "Why this stage is used",
            [
                "EDA checks whether the dataset has enough rows, stores, departments, and time coverage for forecasting.",
                "It reveals missing values, outliers, sales spikes, holiday effects, and seasonality.",
                "It helps decide which features need cleaning, encoding, or transformation before modeling.",
                "It gives reviewer-friendly evidence that the modeling choices are based on observed data behavior.",
            ],
        ),
        (
            "How it is used",
            [
                "The notebook checks date range, row count, unique stores, unique departments, and sales distribution.",
                "Missing values are inspected, especially in markdown and external feature columns.",
                "Sales patterns are visualized over time to show trend and seasonality.",
                "Store-level and department-level behavior is reviewed to confirm that demand differs across groups.",
            ],
        ),
        (
            "Features supported",
            [
                "Descriptive statistics for numeric fields such as Weekly_Sales, Size, CPI, and Unemployment.",
                "Category checks for Store, Dept, Type, and holiday flags.",
                "Time checks using Date to confirm chronological order and validation planning.",
                "Visual evidence later reused in the Streamlit Results Dashboard.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "EDA proves that forecasting is meaningful because sales vary over time, store, department, and holiday periods.",
                "The project does not jump directly to modeling; it first validates the data quality and patterns.",
                "The plots support the explanation of why time-aware and lag-based features are needed.",
            ],
        ),
    ],
    "Feature Engineering": [
        (
            "Why this stage is used",
            [
                "Raw dates and categories are not enough for strong forecasting, so useful signals are extracted.",
                "Calendar features help the model learn seasonal demand patterns.",
                "Holiday, festival, and monsoon flags expose special periods where customer behavior may change.",
                "Feature engineering makes the model more explainable during review because each feature has a business meaning.",
            ],
        ),
        (
            "How it is used",
            [
                "Date is converted into Year, Month, WeekOfYear, and IsMonthEnd.",
                "IsHoliday is kept as an event indicator from the Walmart dataset.",
                "Indian_Festival and Is_Indian_Festival are added as contextual festival features.",
                "Is_Monsoon is created to represent seasonal demand context.",
            ],
        ),
        (
            "Features supported",
            [
                "Calendar features: Year, Month, WeekOfYear, IsMonthEnd.",
                "Event features: IsHoliday, Indian_Festival, Is_Indian_Festival.",
                "Seasonal feature: Is_Monsoon.",
                "Categorical support: Store, Dept, Type, and Indian_Festival can be handled by the model pipeline.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "These features translate business timing into model-readable signals.",
                "The model can learn that sales are not only based on store identity, but also on when the week occurs.",
                "Festival and monsoon features are an added contextual improvement beyond the basic dataset columns.",
            ],
        ),
    ],
    "Lag and Rolling Features": [
        (
            "Why this stage is used",
            [
                "Sales forecasting depends heavily on recent history, so past sales are used as predictive signals.",
                "Lag features tell the model what happened in previous weeks for the same store and department.",
                "Rolling features summarize short-term average demand and volatility.",
                "Using only earlier observations helps avoid data leakage from the future.",
            ],
        ),
        (
            "How it is used",
            [
                "Records are sorted by Store, Dept, and Date before lag features are created.",
                "Sales_Lag_1 captures the previous week of sales.",
                "Sales_Lag_4 and Sales_Lag_12 capture monthly-like and quarter-like sales memory.",
                "Sales_RollMean_4 and Sales_RollStd_12 summarize recent level and variation.",
            ],
        ),
        (
            "Features supported",
            [
                "Sales_Lag_1 for immediate recent demand.",
                "Sales_Lag_4 for short-cycle demand patterns.",
                "Sales_Lag_12 for longer seasonal memory.",
                "Sales_RollMean_4 and Sales_RollStd_12 for trend and volatility.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "This is the main reason the model behaves like a forecasting system, not just a normal tabular regression.",
                "Lag features give the model memory while still respecting time order.",
                "Rolling statistics smooth noisy sales behavior and help the model understand demand stability.",
            ],
        ),
    ],
    "Time-Based Split": [
        (
            "Why this stage is used",
            [
                "Forecasting must imitate real life: train on the past and validate on the future.",
                "A random split would mix future weeks into training and make the model look better than it really is.",
                "Time-based validation gives a more honest estimate of future forecasting performance.",
                "It makes the results easier to defend in a project review.",
            ],
        ),
        (
            "How it is used",
            [
                "Earlier records are assigned to training data.",
                "2012 records are held out as the validation period.",
                "Models learn patterns only from the training period.",
                "Predictions are evaluated on the later validation period using the saved metrics.",
            ],
        ),
        (
            "Features supported",
            [
                "Chronological splitting using Date.",
                "Validation on future weeks rather than shuffled rows.",
                "Leakage control for lag and rolling features.",
                "Comparable evaluation across CatBoost, LightGBM, weighted ensemble, and stacking.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "The split strategy is important because this is a forecasting project.",
                "The model is judged on future-like records, which is closer to deployment behavior.",
                "This makes the reported MAE, RMSE, and R2 more meaningful.",
            ],
        ),
    ],
    "Base Model Training": [
        (
            "Why this stage is used",
            [
                "CatBoost and LightGBM are strong models for structured tabular data.",
                "Using two different gradient boosting learners creates diversity for the later hybrid model.",
                "CatBoost is useful for categorical-heavy data.",
                "LightGBM is fast and often performs very well on large tabular datasets.",
            ],
        ),
        (
            "How it is used",
            [
                "The engineered training features are passed into both base models.",
                "CatBoost receives categorical feature handling for fields such as Store, Dept, Type, and Indian_Festival.",
                "LightGBM uses an ordinal encoder for categorical columns before training.",
                "Each model produces validation predictions that are reused by the ensemble and stacking steps.",
            ],
        ),
        (
            "Features supported",
            [
                "Numerical inputs such as Size, calendar values, lag sales, and rolling statistics.",
                "Categorical inputs such as Store, Dept, Type, and Indian_Festival.",
                "Binary flags such as IsHoliday, IsMonthEnd, Is_Indian_Festival, and Is_Monsoon.",
                "Nonlinear interactions, for example holiday effects differing by store or department.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "These base models are the foundation of the hybrid approach.",
                "The project uses model diversity: CatBoost and LightGBM learn similar data in slightly different ways.",
                "The later hybrid layer depends on these base predictions.",
            ],
        ),
    ],
    "MAE-Weighted Ensemble": [
        (
            "Why this stage is used",
            [
                "It directly implements the report's ensemble idea using validation error.",
                "Models with lower MAE receive higher influence in the final weighted prediction.",
                "It is simple to explain because weights are based on observed model error.",
                "It provides a baseline to compare against the learned stacking hybrid model.",
            ],
        ),
        (
            "How it is used",
            [
                "Validation MAE is calculated for CatBoost and LightGBM.",
                "Inverse-MAE logic gives more weight to the lower-error model.",
                "The weighted prediction is computed from the base model predictions.",
                "Its metrics are saved and displayed beside the individual models and stacking model.",
            ],
        ),
        (
            "Features supported",
            [
                "Uses CatBoost prediction as one ensemble input.",
                "Uses LightGBM prediction as the second ensemble input.",
                "Supports transparent comparison because the weight logic is fixed.",
                "Works as an explainable bridge between single models and stacking.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "This is the report-aligned ensemble baseline.",
                "The method is explainable, but it cannot learn complex relationships between model predictions.",
                "That limitation motivates the next step: the Ridge stacking model.",
            ],
        ),
    ],
    "Stacking Hybrid Model": [
        (
            "Why this stage is used",
            [
                "Stacking upgrades the fixed ensemble into a learned hybrid model.",
                "The meta-learner can learn how much to trust each base model from validation predictions.",
                "Ridge Regression is simple, stable, and easier to defend than a very complex meta-model.",
                "It gives the project a clear hybrid contribution beyond training separate models.",
            ],
        ),
        (
            "How it is used",
            [
                "CatBoost predictions and LightGBM predictions become two input columns for the meta-learner.",
                "Ridge Regression is trained to map those base predictions to actual Weekly_Sales.",
                "During prediction, both base models predict first, then Ridge produces the final forecast.",
                "The learned coefficients are saved and shown in the Hybrid Model Explanation page.",
            ],
        ),
        (
            "Features supported",
            [
                "Meta-input 1: CatBoost prediction.",
                "Meta-input 2: LightGBM prediction.",
                "Learned coefficients show the relative influence of each base model.",
                "The saved artifact contains the base models, encoder, meta-model, feature columns, and categorical columns.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "The hybrid model is not just averaging; it learns the final combination.",
                "This keeps the report's ensemble direction but makes it more adaptive.",
                "A useful limitation to mention is that production stacking should ideally use out-of-fold time-series predictions.",
            ],
        ),
    ],
    "Evaluation": [
        (
            "Why this stage is used",
            [
                "Evaluation proves whether the trained models actually forecast well on held-out future data.",
                "Multiple metrics give a fuller picture than a single score.",
                "Comparing all models shows whether the hybrid approach adds value over base learners.",
                "Saved charts make the result easier to explain visually.",
            ],
        ),
        (
            "How it is used",
            [
                "MAE measures average absolute forecasting error.",
                "MSE penalizes larger errors more strongly.",
                "RMSE gives an error value in the same unit as sales.",
                "R2 shows how much variation in weekly sales is explained by the model.",
            ],
        ),
        (
            "Features supported",
            [
                "Model comparison table for CatBoost, LightGBM, MAE-weighted ensemble, and stacking hybrid.",
                "Actual vs predicted plot to show fit quality.",
                "Error distribution plot to inspect residual behavior.",
                "Sales over time plot to compare forecast movement with actual sales.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "LightGBM has the best MAE in the saved results, while the hybrid model remains close and demonstrates the stacking extension.",
                "Evaluation is done on a time-based validation period, so the metrics are tied to the forecasting setup.",
                "The plots make the numeric results easier for reviewers to understand.",
            ],
        ),
    ],
    "Deployment": [
        (
            "Why this stage is used",
            [
                "Deployment turns the notebook model into an interactive application.",
                "The Streamlit app lets reviewers inspect methodology, results, and live prediction from one place.",
                "Loading a saved joblib artifact proves that the trained model can be reused outside the notebook.",
                "The UI supports project presentation, not only technical experimentation.",
            ],
        ),
        (
            "How it is used",
            [
                "The model artifact is loaded from source/hybrid_stacking_model.joblib.",
                "The Results Dashboard reads saved metrics, coefficient JSON, comparison CSV, and generated plots.",
                "The Prediction Demo collects one feature set from form inputs.",
                "The app runs CatBoost and LightGBM first, encodes categorical values for LightGBM, and then calls the Ridge meta-model.",
            ],
        ),
        (
            "Features supported",
            [
                "Sidebar navigation for summary, walkthrough, report alignment, pipeline, results, explanation, prediction, and viva.",
                "Presentation Mode for larger text and expanded walkthrough details.",
                "Live prediction using Store, Dept, Type, Size, calendar flags, festival, monsoon, lag, and rolling features.",
                "Graceful artifact warnings if model, metrics, CSV, or images are missing.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "The deployment page proves that the ML work is usable, not only written in a notebook.",
                "The UI is designed for defense: it explains the problem, methodology, model, metrics, and limitations.",
                "The final prediction flow mirrors the saved hybrid architecture.",
            ],
        ),
    ],
}


TRAINING_STEPS = [
    "Load dataset",
    "Merge sales, stores, and external features",
    "Clean and inspect data",
    "Create calendar and seasonality features",
    "Add Indian festival and monsoon features",
    "Create lag and rolling sales features",
    "Split data using time-based validation",
    "Train CatBoost",
    "Train LightGBM",
    "Generate base model predictions",
    "Create MAE-weighted ensemble",
    "Train Ridge meta-learner for stacking hybrid model",
    "Evaluate all models",
    "Save final model as joblib",
    "Load model in Streamlit for live prediction",
]


REPORT_COMPARISON = [
    ("Dataset: Walmart sales dataset", "Kaggle Walmart dataset", "Matched"),
    ("Models: XGBoost, LightGBM, CatBoost", "CatBoost + LightGBM", "Partially matched"),
    ("Ensemble: MAE-weighted voting", "MAE-weighted voting implemented", "Matched"),
    ("Hybrid Extension: Not deeply trained", "Stacking meta-learner added", "Improved"),
    ("Evaluation: MAE, MSE, RMSE, R2", "Same metrics used", "Matched"),
    ("Deployment: Not covered", "Streamlit UI added", "Added value"),
]


METRIC_DETAILS = [
    (
        "MAE",
        "MAE - Mean Absolute Error",
        True,
        [
            (
                "Definition",
                [
                    "MAE is the average absolute difference between actual weekly sales and predicted weekly sales.",
                    "It ignores direction, so over-predictions and under-predictions are treated equally.",
                    "Because it is in the same unit as Weekly_Sales, it is usually the easiest error metric to explain.",
                ],
            ),
            (
                "How to read it",
                [
                    "Lower MAE is better.",
                    "A MAE of about 1,414 means the model is off by about 1,414 sales units on average.",
                    "Use this as the main reviewer-friendly metric because it directly describes average forecasting error.",
                ],
            ),
        ],
    ),
    (
        "MSE",
        "MSE - Mean Squared Error",
        True,
        [
            (
                "Definition",
                [
                    "MSE is the average of squared prediction errors.",
                    "Squaring makes large mistakes count much more heavily than small mistakes.",
                    "Its unit is sales squared, so it is useful for comparison but less intuitive to explain as a business number.",
                ],
            ),
            (
                "How to read it",
                [
                    "Lower MSE is better.",
                    "A lower MSE means the model has fewer large forecasting misses.",
                    "Use it to show whether one model avoids big errors better than another model.",
                ],
            ),
        ],
    ),
    (
        "RMSE",
        "RMSE - Root Mean Squared Error",
        True,
        [
            (
                "Definition",
                [
                    "RMSE is the square root of MSE.",
                    "It keeps the large-error penalty from MSE but converts the result back into the sales unit.",
                    "It is usually higher than MAE because it punishes large misses more strongly.",
                ],
            ),
            (
                "How to read it",
                [
                    "Lower RMSE is better.",
                    "A RMSE around 3,099 means larger errors increase the typical error scale to around 3,099 sales units.",
                    "Use it when you want to discuss how strongly the model is affected by occasional big forecasting errors.",
                ],
            ),
        ],
    ),
    (
        "R2",
        "R2 - Coefficient of Determination",
        False,
        [
            (
                "Definition",
                [
                    "R2 measures how much variation in actual weekly sales is explained by the model.",
                    "A value closer to 1 means the predictions explain more of the real sales pattern.",
                    "A value near 0 would mean the model is not much better than predicting an average.",
                ],
            ),
            (
                "How to read it",
                [
                    "Higher R2 is better.",
                    "Here, values around 0.98 show that the models explain most of the variation in weekly sales.",
                    "Use R2 to support the statement that the model captures the overall sales structure strongly.",
                ],
            ),
        ],
    ),
]


PLOT_DETAILS = {
    "Actual vs Predicted": [
        (
            "What this diagram signifies",
            [
                "Each point compares one actual weekly sales value with the model's predicted weekly sales value.",
                "The x-axis is actual sales and the y-axis is predicted sales.",
                "The diagonal line represents perfect prediction, where predicted sales equal actual sales.",
            ],
        ),
        (
            "How it signifies model quality",
            [
                "Points close to the diagonal line mean the model is predicting accurately.",
                "A tight diagonal cloud means the model captures the sales relationship well.",
                "Points far above or below the line are larger errors and usually represent unusual stores, departments, or weeks.",
                "The wider spread at high sales values shows that high-volume sales weeks are harder to predict perfectly.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "The scatter follows the diagonal strongly, so predicted sales generally move with actual sales.",
                "The model is not perfect, but the pattern shows strong fit across low, medium, and high sales ranges.",
                "This visual supports the high R2 shown in the metric table.",
            ],
        ),
    ],
    "Error Distribution": [
        (
            "What this diagram signifies",
            [
                "This histogram shows prediction errors across the validation data.",
                "Errors near zero mean the prediction was close to the actual weekly sales value.",
                "The x-axis is prediction error and the y-axis is the number of records with that error range.",
            ],
        ),
        (
            "How it signifies model quality",
            [
                "The tallest bars near zero mean most predictions have small errors.",
                "A distribution centered near zero suggests the model does not have a strong overall bias.",
                "Long tails on either side represent rare cases where the model made much larger mistakes.",
                "A narrower distribution would mean more stable predictions; wider tails show harder edge cases.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "Most errors are concentrated close to zero, which supports the low MAE and RMSE values.",
                "The few extreme errors are expected in retail data because holidays, promotions, or unusual demand spikes can be difficult.",
                "This plot helps explain not only average error, but also the spread of errors.",
            ],
        ),
    ],
    "Sales Over Time": [
        (
            "What this diagram signifies",
            [
                "This line chart compares actual total weekly sales and predicted total weekly sales over the validation period.",
                "The blue line shows actual sales movement over time.",
                "The orange line shows the stacking hybrid model's predicted sales movement over time.",
            ],
        ),
        (
            "How it signifies model quality",
            [
                "When both lines rise and fall together, the model is capturing time-based sales patterns.",
                "Small gaps between the lines mean the forecast is close for that week.",
                "Large gaps show weeks where the model missed a peak, dip, or unusual demand event.",
                "This plot is useful because forecasting should be judged across time, not only row-by-row metrics.",
            ],
        ),
        (
            "What to say while presenting",
            [
                "The predicted line follows the actual line closely, so the model captures the overall validation-period trend.",
                "Some weekly peaks and dips are not matched perfectly, which is realistic for retail forecasting.",
                "This visual proves the model is not only accurate in aggregate metrics, but also follows the time pattern.",
            ],
        ),
    ],
}


MODEL_EXPLANATION_DETAILS = [
    (
        "CatBoost",
        [
            (
                "Definition",
                [
                    "CatBoost is a gradient boosting decision-tree model designed to work well with categorical features.",
                    "It builds many small decision trees in sequence, where each new tree corrects mistakes made by earlier trees.",
                    "It is often strong on business datasets containing mixed numeric and categorical columns.",
                ],
            ),
            (
                "How it helps in this project",
                [
                    "It can learn patterns from Store, Dept, Type, Indian_Festival, and numeric sales-history features.",
                    "It captures nonlinear relationships such as holiday impact being different for different departments.",
                    "It acts as one base learner whose prediction is later used by the ensemble and stacking hybrid model.",
                ],
            ),
        ],
    ),
    (
        "LightGBM",
        [
            (
                "Definition",
                [
                    "LightGBM is a fast gradient boosting decision-tree model optimized for large tabular datasets.",
                    "It grows trees efficiently and can capture complex nonlinear relationships without needing manual formulas.",
                    "In this project, categorical columns are encoded before being passed into LightGBM.",
                ],
            ),
            (
                "How it helps in this project",
                [
                    "It performs strongly on the merged Walmart sales table and has the best MAE in the saved results.",
                    "It learns interactions between lag sales, rolling statistics, calendar features, and store attributes.",
                    "It gives a second base prediction that is complementary to CatBoost for the hybrid layer.",
                ],
            ),
        ],
    ),
    (
        "Ridge Regression Meta-Learner",
        [
            (
                "Definition",
                [
                    "Ridge Regression is linear regression with L2 regularization.",
                    "The regularization controls coefficient size, which helps keep the final blending model stable.",
                    "It is simple and interpretable, making it a good meta-learner for project defense.",
                ],
            ),
            (
                "How it helps in this project",
                [
                    "It takes CatBoost and LightGBM predictions as inputs and learns the final sales forecast.",
                    "It learns a correction-style blend instead of using manually chosen weights.",
                    "Its coefficients can be shown to reviewers to explain how the hybrid model combines base predictions.",
                ],
            ),
        ],
    ),
    (
        "MAE-Weighted Ensemble",
        [
            (
                "Definition",
                [
                    "The MAE-weighted ensemble is a fixed weighted average of CatBoost and LightGBM predictions.",
                    "The model with lower validation MAE receives a larger influence in the final prediction.",
                    "It is not a trained meta-model; it is a transparent error-based voting method.",
                ],
            ),
            (
                "How it helps in this project",
                [
                    "It directly matches the report's weighted-voting ensemble idea.",
                    "It provides an explainable baseline between individual models and the learned stacking model.",
                    "It shows whether simple error-based weighting is competitive with the trained hybrid approach.",
                ],
            ),
        ],
    ),
    (
        "Stacking Hybrid Model",
        [
            (
                "Definition",
                [
                    "A stacking model trains base models first, then trains a second-level model on their predictions.",
                    "Here, CatBoost and LightGBM are the base models, and Ridge Regression is the second-level model.",
                    "The final forecast is learned from model outputs rather than from a fixed average only.",
                ],
            ),
            (
                "How it helps in this project",
                [
                    "It turns the report's ensemble idea into a true trained hybrid architecture.",
                    "It can learn when one base model should influence the final forecast more than the other.",
                    "It gives a clear final contribution: base learners plus a learned meta-learner deployed in Streamlit.",
                ],
            ),
        ],
    ),
    (
        "Detailed Hybrid Prediction Flow",
        [
            (
                "Step-by-step flow",
                [
                    "The user or validation row provides store, department, calendar, holiday, festival, monsoon, lag, and rolling features.",
                    "CatBoost predicts weekly sales from the feature row.",
                    "Categorical columns are encoded for LightGBM, then LightGBM predicts weekly sales from the same feature row.",
                    "The two predictions are placed side by side as meta-features.",
                    "Ridge Regression uses those two meta-features to produce the final stacking hybrid forecast.",
                ],
            ),
            (
                "How to explain the coefficients",
                [
                    "The coefficients are not simple percentage weights; they are learned correction terms inside Ridge Regression.",
                    "A larger LightGBM coefficient matches the result table, where LightGBM is the strongest individual model.",
                    "A negative CatBoost coefficient does not mean CatBoost is useless; it means Ridge learned to use CatBoost as a correction signal in the final blend.",
                    "The intercept is a small adjustment added after combining the two base predictions.",
                ],
            ),
            (
                "Important limitation to mention",
                [
                    "For a production-grade stacking system, the meta-learner should ideally be trained using out-of-fold time-series predictions.",
                    "That avoids giving the meta-learner overly easy base predictions from the same training period.",
                    "This can be added later using TimeSeriesSplit as future work.",
                ],
            ),
        ],
    ),
]


PREDICTION_FIELD_HELP = {
    "Store": "Store number from the Walmart dataset. It identifies the branch whose weekly sales are being forecasted.",
    "Dept": "Department number inside the selected store. Demand differs strongly by department.",
    "Type": "Store category A, B, or C. This helps the model learn differences between store formats.",
    "Size": "Physical store size. Larger stores usually have different sales capacity and customer volume.",
    "Year": "Forecast year used as a calendar feature.",
    "Month": "Forecast month. This captures seasonal retail behavior.",
    "WeekOfYear": "Week number in the year. This helps the model learn weekly seasonality.",
    "IsHoliday": "Marks whether the week is a holiday week. Holiday weeks can create spikes or drops in demand.",
    "IsMonthEnd": "Marks whether the date is near month end, where shopping behavior may shift.",
    "Indian_Festival": "Contextual festival indicator used to demonstrate event-aware retail forecasting.",
    "Is_Indian_Festival": "Automatically becomes 1 when a festival is selected and 0 when None is selected.",
    "Is_Monsoon": "Seasonal flag that represents monsoon-period context for demand changes.",
    "Sales_Lag_1": "Previous week sales for the same Store and Dept. This is the strongest short-term memory signal.",
    "Sales_Lag_4": "Sales from four weeks earlier. This captures monthly-like demand memory.",
    "Sales_Lag_12": "Sales from twelve weeks earlier. This gives the model longer seasonal memory.",
    "Sales_RollMean_4": "Average sales over the recent four-week window. It smooths short-term noise.",
    "Sales_RollStd_12": "Sales volatility over a twelve-week window. It tells the model whether demand is stable or unstable.",
}


def source_label(filename: str) -> str:
    return f"source/{filename}"


def apply_css(presentation_mode: bool) -> None:
    presentation_css = ""
    if presentation_mode:
        presentation_css = """
        .stMarkdown p, .stMarkdown li, .stExpander p, .stExpander li {
            font-size: 1.08rem !important;
            line-height: 1.7 !important;
        }
        h1 { font-size: 3.2rem !important; }
        h2 { font-size: 2.1rem !important; }
        h3 { font-size: 1.55rem !important; }
        """

    st.markdown(
        f"""
        <style>
        :root {{
            --navy: #17202f;
            --ink: #223047;
            --muted: #5f6c7b;
            --blue: #2563eb;
            --cyan: #0891b2;
            --green: #16a34a;
            --amber: #d97706;
            --red: #dc2626;
            --panel: #ffffff;
            --soft: #eef6ff;
            --line: #dbe5f0;
            --page: #f5f8fc;
            --input: #ffffff;
            --shadow: 0 14px 34px rgba(23, 32, 47, 0.08);
        }}

        html, body, .stApp, [data-testid="stAppViewContainer"] {{
            background: var(--page) !important;
            color: var(--ink) !important;
        }}

        [data-testid="stHeader"] {{
            background: rgba(245, 248, 252, 0.94) !important;
        }}

        [data-testid="stSidebar"] {{
            background: #ffffff !important;
            border-right: 1px solid var(--line);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--ink) !important;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            background: #f8fbff !important;
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 0.35rem 0.45rem;
            margin-bottom: 0.12rem;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: #eef6ff !important;
            border-color: #bfdbfe;
        }}

        [data-testid="stSidebar"] code {{
            background: #eef6ff !important;
            color: #174ea6 !important;
            border: 1px solid #bfdbfe;
        }}

        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }}

        h1, h2, h3,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] strong,
        label,
        [data-testid="stWidgetLabel"] p {{
            color: var(--navy);
            letter-spacing: 0;
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {{
            color: var(--ink);
        }}

        [data-testid="stForm"],
        [data-testid="stMetric"],
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {{
            background: #ffffff !important;
            border: 1px solid var(--line);
            border-radius: 8px;
        }}

        [data-testid="stForm"] {{
            padding: 1.15rem 1.2rem 1.25rem 1.2rem;
            box-shadow: var(--shadow);
            background:
                linear-gradient(180deg, rgba(239, 246, 255, 0.72), rgba(255, 255, 255, 0.96)),
                #ffffff !important;
        }}

        [data-testid="stMetric"] {{
            padding: 0.8rem;
        }}

        [data-testid="stMetric"] label,
        [data-testid="stMetric"] div {{
            color: var(--ink) !important;
        }}

        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="popover"] {{
            background: var(--input) !important;
            color: var(--ink) !important;
            border-color: #cbd5e1 !important;
        }}

        div[data-baseweb="input"]:hover,
        div[data-baseweb="select"] > div:hover {{
            border-color: var(--blue) !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.08);
        }}

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span {{
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }}

        div[data-baseweb="select"] svg,
        div[data-baseweb="checkbox"] svg {{
            color: var(--blue) !important;
            fill: var(--blue) !important;
        }}

        button[kind="primary"] {{
            background: var(--blue) !important;
            border: 1px solid var(--blue) !important;
            color: #ffffff !important;
            font-weight: 850 !important;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.2);
        }}

        button[kind="primary"]:hover {{
            background: #1d4ed8 !important;
            border-color: #1d4ed8 !important;
            transform: translateY(-1px);
        }}

        button[kind="secondary"] {{
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            color: var(--ink) !important;
        }}

        [data-testid="stAlert"] {{
            background: #fff7ed !important;
            color: #7c2d12 !important;
            border: 1px solid #fed7aa;
            border-radius: 8px;
        }}

        [data-testid="stAlert"] * {{
            color: inherit !important;
        }}

        .main-hero {{
            border: 1px solid var(--line);
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.13), rgba(8, 145, 178, 0.10)),
                #ffffff;
            padding: 1.45rem 1.55rem;
            border-radius: 8px;
            box-shadow: 0 12px 32px rgba(23, 32, 47, 0.08);
            margin-bottom: 1rem;
        }}

        .hero-kicker {{
            color: var(--cyan);
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }}

        .hero-title {{
            color: var(--navy);
            font-size: clamp(2rem, 4vw, 3.35rem);
            line-height: 1.04;
            font-weight: 900;
            margin: 0 0 0.55rem 0;
        }}

        .hero-copy {{
            color: var(--ink);
            font-size: 1.04rem;
            line-height: 1.65;
            max-width: 920px;
            margin: 0;
        }}

        .mini-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 0.8rem;
            margin: 1rem 0;
        }}

        .metric-card, .info-card, .flow-card, .step-card, .say-box,
        .demo-card, .derived-card, .viva-card, .artifact-card {{
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(23, 32, 47, 0.06);
        }}

        .metric-card:hover, .info-card:hover, .flow-card:hover, .step-card:hover,
        .demo-card:hover, .derived-card:hover, .viva-card:hover, .artifact-card:hover {{
            border-color: #bfdbfe;
            box-shadow: 0 14px 28px rgba(23, 32, 47, 0.09);
            transform: translateY(-1px);
            transition: all 140ms ease;
        }}

        .metric-card {{
            min-height: 116px;
        }}

        .metric-label {{
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        .metric-value {{
            color: var(--navy);
            font-size: 1.65rem;
            line-height: 1.2;
            font-weight: 900;
            margin-top: 0.25rem;
        }}

        .metric-note {{
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 0.25rem;
        }}

        .pill-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.65rem 0 0.1rem 0;
        }}

        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #1d4ed8;
            padding: 0.34rem 0.62rem;
            font-size: 0.84rem;
            font-weight: 800;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 800;
            white-space: nowrap;
        }}

        .badge-green {{ background: #dcfce7; color: #166534; }}
        .badge-blue {{ background: #dbeafe; color: #1d4ed8; }}
        .badge-amber {{ background: #fef3c7; color: #92400e; }}

        .flow-card {{
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 0.85rem;
            align-items: start;
            margin: 0.45rem 0;
            border-left: 5px solid var(--blue);
        }}

        .flow-icon {{
            width: 46px;
            height: 46px;
            border-radius: 8px;
            background: #eff6ff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.45rem;
        }}

        .flow-title {{
            color: var(--navy);
            font-size: 1.08rem;
            font-weight: 900;
            margin-bottom: 0.1rem;
        }}

        .flow-subtitle {{
            color: var(--muted);
            font-size: 0.9rem;
            font-weight: 700;
        }}

        .arrow {{
            color: var(--cyan);
            text-align: center;
            font-size: 1.45rem;
            font-weight: 900;
            margin: 0.15rem 0;
        }}

        .say-box {{
            background: linear-gradient(90deg, #ecfeff, #ffffff);
            border-left: 5px solid var(--cyan);
            margin: 0.85rem 0 1.15rem 0;
        }}

        .say-title {{
            color: #155e75;
            font-weight: 900;
            margin-bottom: 0.25rem;
        }}

        .say-text {{
            color: var(--ink);
            font-size: 0.98rem;
            line-height: 1.55;
        }}

        .comparison-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(23, 32, 47, 0.06);
        }}

        .comparison-table th {{
            background: #17202f;
            color: #ffffff;
            text-align: left;
            padding: 0.72rem 0.8rem;
            font-size: 0.86rem;
        }}

        .comparison-table td {{
            border-top: 1px solid var(--line);
            color: var(--ink);
            padding: 0.72rem 0.8rem;
            vertical-align: top;
            font-size: 0.92rem;
        }}

        .step-card {{
            min-height: 112px;
            border-top: 4px solid var(--cyan);
        }}

        .step-num {{
            color: var(--cyan);
            font-weight: 900;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        .step-text {{
            color: var(--navy);
            font-weight: 850;
            font-size: 1rem;
            margin-top: 0.35rem;
        }}

        .soft-band {{
            border: 1px solid var(--line);
            background: #f8fbff;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.9rem 0;
        }}

        .form-section-title {{
            border-left: 4px solid var(--blue);
            background: rgba(239, 246, 255, 0.92);
            border-radius: 8px;
            padding: 0.72rem 0.88rem;
            margin: 1rem 0 0.7rem 0;
            color: var(--navy);
            font-weight: 900;
        }}

        .form-section-title span {{
            display: block;
            color: var(--muted);
            font-size: 0.84rem;
            font-weight: 700;
            margin-top: 0.18rem;
        }}

        .demo-grid, .viva-grid, .artifact-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.85rem 0 1rem 0;
        }}

        .demo-card {{
            position: relative;
            min-height: 120px;
            border-top: 4px solid var(--cyan);
        }}

        .demo-label, .viva-label, .artifact-label {{
            color: var(--cyan);
            font-size: 0.76rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.34rem;
        }}

        .demo-title, .viva-title, .artifact-title {{
            color: var(--navy);
            font-size: 1.08rem;
            font-weight: 900;
            line-height: 1.25;
        }}

        .demo-text, .viva-text, .artifact-text {{
            color: var(--ink);
            font-size: 0.93rem;
            line-height: 1.5;
            margin-top: 0.35rem;
        }}

        .derived-card {{
            min-height: 104px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            background: #f8fbff;
        }}

        .derived-value {{
            color: var(--navy);
            font-size: 2rem;
            line-height: 1.1;
            font-weight: 950;
            margin-top: 0.28rem;
        }}

        .hover-tip {{
            position: relative;
            cursor: help;
        }}

        .hover-tip::after {{
            content: attr(data-tip);
            position: absolute;
            left: 0;
            bottom: calc(100% + 10px);
            min-width: 230px;
            max-width: 300px;
            background: #111827;
            color: #ffffff;
            border-radius: 8px;
            padding: 0.58rem 0.7rem;
            font-size: 0.8rem;
            line-height: 1.35;
            opacity: 0;
            pointer-events: none;
            transform: translateY(4px);
            transition: all 130ms ease;
            z-index: 9999;
            box-shadow: 0 10px 24px rgba(17, 24, 39, 0.18);
        }}

        .hover-tip:hover::after {{
            opacity: 1;
            transform: translateY(0);
        }}

        .prediction-result {{
            border: 1px solid #bbf7d0;
            background: linear-gradient(90deg, #ecfdf5, #ffffff);
            border-left: 5px solid var(--green);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 0 12px 28px rgba(22, 163, 74, 0.09);
        }}

        .prediction-result small {{
            color: #166534;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }}

        .prediction-result strong {{
            display: block;
            color: var(--navy);
            font-size: 2rem;
            margin-top: 0.2rem;
        }}

        .viva-card {{
            border-top: 4px solid var(--blue);
            min-height: 138px;
        }}

        .artifact-grid {{
            grid-template-columns: repeat(auto-fit, minmax(245px, 1fr));
        }}

        .artifact-card {{
            display: flex;
            gap: 0.85rem;
            align-items: flex-start;
        }}

        .artifact-status {{
            flex: 0 0 auto;
            border-radius: 999px;
            padding: 0.25rem 0.58rem;
            font-size: 0.72rem;
            font-weight: 900;
            text-transform: uppercase;
        }}

        .artifact-ok {{
            background: #dcfce7;
            color: #166534;
        }}

        .artifact-missing {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .artifact-path {{
            color: #166534;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.82rem;
            overflow-wrap: anywhere;
            margin-top: 0.18rem;
        }}

        .summary-slide {{
            border: 1px solid var(--line);
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.08), rgba(8, 145, 178, 0.08)),
                #ffffff;
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 14px 34px rgba(23, 32, 47, 0.08);
            margin-top: 0.8rem;
        }}

        .summary-head {{
            display: grid;
            grid-template-columns: 1.3fr 0.7fr;
            gap: 1rem;
            align-items: stretch;
            margin-bottom: 1rem;
        }}

        .summary-title {{
            color: var(--navy);
            font-size: clamp(1.65rem, 3.2vw, 2.7rem);
            font-weight: 950;
            line-height: 1.08;
            margin-bottom: 0.45rem;
        }}

        .summary-subtitle {{
            color: var(--ink);
            font-size: 1rem;
            line-height: 1.55;
            max-width: 760px;
        }}

        .summary-badge {{
            border-radius: 8px;
            background: var(--navy);
            color: #ffffff;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 130px;
        }}

        .summary-badge small {{
            color: #bfdbfe;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 900;
        }}

        .summary-badge strong {{
            color: #ffffff;
            font-size: 1.45rem;
            margin-top: 0.35rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 1rem 0;
        }}

        .summary-card {{
            border: 1px solid var(--line);
            background: #ffffff;
            border-radius: 8px;
            padding: 0.95rem;
            min-height: 145px;
        }}

        .summary-card h4 {{
            color: var(--cyan);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 0 0 0.4rem 0;
        }}

        .summary-card p {{
            color: var(--ink);
            font-size: 0.95rem;
            line-height: 1.55;
            margin: 0;
        }}

        .summary-footer {{
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #174ea6;
            border-radius: 8px;
            padding: 0.95rem 1rem;
            font-weight: 850;
            line-height: 1.55;
            margin-top: 0.9rem;
        }}

        @media (max-width: 860px) {{
            .summary-head,
            .summary-grid,
            .demo-grid,
            .viva-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        div[data-testid="stExpander"] {{
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff !important;
            box-shadow: 0 8px 20px rgba(23, 32, 47, 0.05);
            margin-bottom: 0.58rem;
        }}

        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary {{
            background: #ffffff !important;
            color: var(--ink) !important;
            border-radius: 8px;
        }}

        div[data-testid="stExpander"] summary p {{
            font-weight: 850;
            color: var(--navy);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.35rem;
            border-bottom: 1px solid var(--line);
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 0.55rem 0.8rem;
            background: #eaf3ff;
            color: var(--ink);
        }}

        .stTabs [data-baseweb="tab"] p {{
            color: var(--ink) !important;
            font-weight: 800;
        }}

        .stTabs [aria-selected="true"] {{
            background: var(--navy) !important;
        }}

        .stTabs [aria-selected="true"] p {{
            color: #ffffff !important;
        }}

        {presentation_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_json_file(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_comparison_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model_artifact(path_text: str) -> tuple[dict[str, Any] | None, str | None]:
    path = Path(path_text)
    if not path.exists():
        return None, f"Model file missing: {source_label(path.name)}"

    try:
        import joblib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(path), None
    except Exception as exc:
        return None, f"Could not load {source_label(path.name)}. Details: {exc}"


def safe_image(filename: str, caption: str | None = None) -> None:
    path = SOURCE_DIR / filename
    if path.exists():
        st.image(source_label(filename), caption=caption, width="stretch")
    else:
        st.warning(f"Missing image: {source_label(filename)}")


def hero(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <section class="main-hero">
            <div class="hero-kicker">{kicker}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-copy">{copy}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def review_box(text: str) -> None:
    st.markdown(
        f"""
        <div class="say-box">
            <div class="say-title">What to say in review</div>
            <div class="say-text">Say this: {text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def coefficient_card(label: str, value: str, note: str, tip: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card hover-tip" data-tip="{tip}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pills(items: list[str]) -> None:
    pills = "".join(f'<span class="pill">{item}</span>' for item in items)
    st.markdown(f'<div class="pill-row">{pills}</div>', unsafe_allow_html=True)


def status_badge(status: str) -> str:
    cls = {
        "Matched": "badge-green",
        "Partially matched": "badge-amber",
        "Improved": "badge-blue",
        "Added value": "badge-blue",
    }.get(status, "badge-blue")
    return f'<span class="badge {cls}">{status}</span>'


def report_comparison_table() -> None:
    rows = []
    for report_method, implementation, status in REPORT_COMPARISON:
        rows.append(
            "<tr>"
            f"<td>{report_method}</td>"
            f"<td>{implementation}</td>"
            f"<td>{status_badge(status)}</td>"
            "</tr>"
        )

    st.markdown(
        """
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Report Method</th>
                    <th>Our Implementation</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        """
        + "".join(rows)
        + """
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def clean_model_name(name: str) -> str:
    return name.replace("_", " ")


def format_metric_value(metric_name: str, value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not available"

    if metric_name == "R2":
        return f"{number:.4f}"
    return f"{number:,.2f}"


def metric_summary_items(metrics: dict[str, Any], metric_name: str, lower_is_better: bool) -> list[str]:
    if not metrics:
        return ["The saved metrics file is not available, so the project-specific comparison cannot be shown."]

    valid_models = {
        model: values
        for model, values in metrics.items()
        if isinstance(values, dict) and metric_name in values
    }
    if not valid_models:
        return [f"No saved {metric_name} values are available for comparison."]

    selector = min if lower_is_better else max
    best_model = selector(valid_models, key=lambda model: valid_models[model].get(metric_name, 0))
    best_value = valid_models[best_model].get(metric_name)
    hybrid_value = valid_models.get("Stacking_Hybrid_Model", {}).get(metric_name)
    direction = "lowest" if lower_is_better else "highest"

    items = [
        f"Best {metric_name} here is the {direction} value: {clean_model_name(best_model)} at {format_metric_value(metric_name, best_value)}.",
    ]
    if hybrid_value is not None:
        items.append(
            f"The stacking hybrid model records {metric_name} = {format_metric_value(metric_name, hybrid_value)}, so it can be compared directly with the base models and weighted ensemble."
        )
    items.append(
        "Use this metric together with the plots below, because metrics summarize the error while plots show where the model fits or misses."
    )
    return items


def expand_collapse_controls(
    control_key: str,
    default_expanded: bool = False,
    expand_label: str = "Expand all",
    collapse_label: str = "Collapse all",
) -> bool:
    state_key = f"{control_key}_expanded"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_expanded

    cols = st.columns([1.15, 1.15, 5])
    with cols[0]:
        if st.button(expand_label, key=f"{control_key}_expand", use_container_width=True):
            st.session_state[state_key] = True
    with cols[1]:
        if st.button(collapse_label, key=f"{control_key}_collapse", use_container_width=True):
            st.session_state[state_key] = False

    return bool(st.session_state[state_key])


def render_bullet_section(title: str, items: list[str]) -> None:
    st.markdown(f"**{title}**")
    st.markdown("\n".join(f"- {item}" for item in items))


def render_detail_sections(sections: list[tuple[str, list[str]]]) -> None:
    for row_start in range(0, len(sections), 2):
        cols = st.columns(2)
        for offset, (title, items) in enumerate(sections[row_start : row_start + 2]):
            with cols[offset]:
                render_bullet_section(title, items)


def render_detail_expanders(
    details: list[tuple[str, list[tuple[str, list[str]]]]],
    control_key: str,
    default_expanded: bool = False,
) -> None:
    expanded = expand_collapse_controls(control_key, default_expanded)
    for title, sections in details:
        with st.expander(title, expanded=expanded):
            render_detail_sections(sections)


def metric_definition_details(metrics: dict[str, Any]) -> list[tuple[str, list[tuple[str, list[str]]]]]:
    details = []
    for metric_name, title, lower_is_better, sections in METRIC_DETAILS:
        metric_sections = list(sections)
        metric_sections.append(
            (
                "What it signifies in this result table",
                metric_summary_items(metrics, metric_name, lower_is_better),
            )
        )
        details.append((title, metric_sections))
    return details


def render_plot_explanation(plot_name: str, expanded: bool) -> None:
    with st.expander(f"How to read this diagram: {plot_name}", expanded=expanded):
        render_detail_sections(PLOT_DETAILS[plot_name])


def render_step_detail_sections(step: dict[str, Any]) -> None:
    detail_sections = FLOW_STEP_DETAILS.get(str(step["title"]), [])
    if not detail_sections:
        return

    st.markdown("#### Presenter Details")
    render_detail_sections(detail_sections)


def flow_card(step: dict[str, Any], index: int) -> None:
    st.markdown(
        f"""
        <div class="flow-card">
            <div class="flow-icon">{step["icon"]}</div>
            <div>
                <div class="flow-title">{index}. {step["title"]}</div>
                <div class="flow-subtitle">{step["subtitle"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flowchart(expand_all: bool = False, control_key: str = "flowchart") -> None:
    st.progress(1.0, text="Raw dataset to deployed Streamlit forecasting dashboard")
    expanded = expand_collapse_controls(
        control_key,
        expand_all,
        expand_label="Expand all",
        collapse_label="Collapse all",
    )

    for index, step in enumerate(FLOW_STEPS, start=1):
        flow_card(step, index)
        with st.expander(f"Open step {index}: {step['title']}", expanded=expanded):
            st.markdown("#### Quick Explanation")
            cols = st.columns(3)
            with cols[0]:
                st.markdown("**What was done**")
                st.markdown(f"- {step['what']}")
            with cols[1]:
                st.markdown("**Why it was done**")
                st.markdown(f"- {step['why']}")
            with cols[2]:
                st.markdown("**Connection to report**")
                st.markdown(f"- {step['report']}")

            render_step_detail_sections(step)

        if index < len(FLOW_STEPS):
            st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)


def render_training_steps() -> None:
    for row_start in range(0, len(TRAINING_STEPS), 3):
        cols = st.columns(3)
        for offset, step_text in enumerate(TRAINING_STEPS[row_start : row_start + 3]):
            step_number = row_start + offset + 1
            with cols[offset]:
                st.markdown(
                    f"""
                    <div class="step-card">
                        <div class="step-num">Step {step_number}</div>
                        <div class="step-text">{step_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def metrics_dict() -> dict[str, Any]:
    return load_json_file(str(METRICS_PATH))


def meta_dict() -> dict[str, Any]:
    return load_json_file(str(META_PATH))


def comparison_df() -> pd.DataFrame:
    return load_comparison_csv(str(COMPARISON_PATH))


def render_artifact_warnings() -> None:
    expected_files = [
        "hybrid_stacking_model.joblib",
        "metrics.json",
        "meta_coefficients.json",
        "model_comparison_metrics.csv",
        "actual_vs_predicted_hybrid.png",
        "error_distribution_hybrid.png",
        "sales_over_time_hybrid.png",
    ]
    missing = [source_label(name) for name in expected_files if not (SOURCE_DIR / name).exists()]
    if missing:
        st.warning("Missing project files: " + ", ".join(missing))


def page_home(presentation_mode: bool) -> None:
    hero(
        "Hybrid Model UI",
        "Retail Sales Forecasting Defense Dashboard",
        "A Streamlit presentation hub for the research report, methodology, model training pipeline, evaluation results, and live prediction workflow.",
    )

    render_pills(
        [
            "📘 report aligned",
            "🌳 gradient boosting",
            "⚖️ MAE ensemble",
            "🧠 Ridge stacking",
            "🚀 deployed UI",
        ]
    )

    metrics = metrics_dict()
    best_model = "LightGBM"
    if metrics:
        best_model = min(metrics, key=lambda key: metrics[key].get("MAE", float("inf")))

    cols = st.columns(4)
    with cols[0]:
        metric_card("Project Scope", "End-to-end", "Notebook, model, visuals, and UI")
    with cols[1]:
        metric_card("Best MAE Model", best_model.replace("_", " "), "Lower MAE is better")
    with cols[2]:
        metric_card("Hybrid Layer", "Ridge", "Learns model combination")
    with cols[3]:
        metric_card("Deployment", "Streamlit", "Loaded from source/joblib")

    st.markdown("### Presentation Map")
    map_cols = st.columns(3)
    with map_cols[0]:
        st.markdown(
            """
            <div class="info-card">
                <strong>1. Explain alignment</strong><br>
                Show that the implementation follows the report's gradient boosting and MAE ensemble idea.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with map_cols[1]:
        st.markdown(
            """
            <div class="info-card">
                <strong>2. Walk through pipeline</strong><br>
                Move from Kaggle data to feature engineering, training, stacking, evaluation, and deployment.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with map_cols[2]:
        st.markdown(
            """
            <div class="info-card">
                <strong>3. Demonstrate output</strong><br>
                Use results charts and the prediction demo to show that the model is usable.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Key Result Visuals")
    chart_cols = st.columns(3)
    with chart_cols[0]:
        safe_image("actual_vs_predicted_hybrid.png", "Actual vs predicted weekly sales")
    with chart_cols[1]:
        safe_image("error_distribution_hybrid.png", "Hybrid model error distribution")
    with chart_cols[2]:
        safe_image("sales_over_time_hybrid.png", "Actual vs predicted sales over time")

    review_box(
        "This dashboard is not only a predictor. It is a project defense tool that connects the report, methodology, trained hybrid model, metrics, and deployment in one place."
    )
    render_artifact_warnings()


def page_one_slide_summary(presentation_mode: bool) -> None:
    metrics = metrics_dict()
    hybrid = metrics.get("Stacking_Hybrid_Model", {})
    best_model = "LightGBM"
    best_mae = 0.0
    if metrics:
        best_model = min(metrics, key=lambda key: metrics[key].get("MAE", float("inf")))
        best_mae = metrics[best_model].get("MAE", 0.0)

    hybrid_mae = hybrid.get("MAE", 0.0)
    hybrid_r2 = hybrid.get("R2", 0.0)

    st.html(
        f"""
        <section class="summary-slide">
            <div class="summary-head">
                <div>
                    <div class="summary-title">Retail Sales Forecasting Using a Hybrid Stacking Model</div>
                    <div class="summary-subtitle">
                        A complete implementation of the report idea: Kaggle Walmart sales data, gradient boosting base models,
                        MAE-weighted voting baseline, Ridge Regression stacking, evaluation, and Streamlit deployment.
                    </div>
                </div>
                <div class="summary-badge">
                    <small>Final Contribution</small>
                    <strong>Report-aligned hybrid forecasting dashboard</strong>
                </div>
            </div>

            <div class="summary-grid">
                <div class="summary-card">
                    <h4>Problem</h4>
                    <p>Forecast weekly retail sales accurately so store and department-level demand can be understood before future weeks arrive.</p>
                </div>
                <div class="summary-card">
                    <h4>Dataset</h4>
                    <p>Kaggle Walmart dataset using train.csv, stores.csv, and features.csv merged into one modeling table.</p>
                </div>
                <div class="summary-card">
                    <h4>Features</h4>
                    <p>Calendar, holiday, Indian festival, monsoon, lag sales, and rolling sales features capture seasonality and sales memory.</p>
                </div>
                <div class="summary-card">
                    <h4>Models</h4>
                    <p>CatBoost and LightGBM are trained as gradient boosting base learners, matching the report's modeling direction.</p>
                </div>
                <div class="summary-card">
                    <h4>Hybrid Upgrade</h4>
                    <p>The report's MAE-weighted ensemble is implemented, then extended with a Ridge meta-learner for stacking.</p>
                </div>
                <div class="summary-card">
                    <h4>Deployment</h4>
                    <p>The saved joblib model and generated findings are loaded into Streamlit for live prediction and project defense.</p>
                </div>
            </div>

            <div class="summary-footer">
                Key result: best MAE model is {best_model.replace("_", " ")} with MAE {best_mae:,.2f}.
                The stacking hybrid model reports MAE {hybrid_mae:,.2f} and R2 {hybrid_r2:.4f}.
            </div>
        </section>
        """
    )

    st.markdown("### Presentation Closing Line")
    review_box(
        "This project follows the report's gradient boosting and MAE-weighted ensemble approach, then adds a trained stacking hybrid model and deploys it through a Streamlit dashboard for explanation and prediction."
    )


def page_project_walkthrough(presentation_mode: bool) -> None:
    hero(
        "Project Walkthrough",
        "From Research Report to Working Forecasting App",
        "Use this page as the main guided explanation during review. It connects the report's idea to the implemented model and the deployed Streamlit interface.",
    )

    overview_tab, flow_tab, training_tab, review_tab = st.tabs(
        ["Report Fit", "Interactive Flowchart", "Training Steps", "Review Script"]
    )

    with overview_tab:
        st.markdown("### Why the implementation matches the report")
        st.markdown(
            """
            The report proposed retail sales forecasting with gradient boosting models and an MAE-weighted voting ensemble.
            This implementation follows that structure with CatBoost and LightGBM, then extends it with a stacking layer.

            The important defense point is simple: fixed MAE weights are implemented as the report baseline, and the Ridge
            Regression meta-learner is an upgrade that learns the final blend from model predictions.
            """
        )
        report_comparison_table()
        review_box(
            "The implementation follows the report's gradient boosting ensemble idea, then improves it by adding a trained Ridge Regression stacking layer."
        )

    with flow_tab:
        render_flowchart(expand_all=presentation_mode, control_key="walkthrough_flowchart")
        review_box(
            "The full pipeline starts from the Kaggle dataset, builds forecasting features, trains gradient boosting models, compares an MAE-weighted ensemble, and deploys the saved hybrid model in Streamlit."
        )

    with training_tab:
        render_training_steps()
        review_box(
            "The model was trained in a structured sequence: merge data, engineer forecasting features, split by time, train base learners, create ensemble predictions, train the Ridge meta-learner, evaluate, and save the final joblib artifact."
        )

    with review_tab:
        st.markdown(
            """
            <div class="soft-band">
                <strong>Opening:</strong> This project implements retail sales forecasting on the Kaggle Walmart dataset using gradient boosting models.
                The report proposed XGBoost, LightGBM, CatBoost, and MAE-weighted voting. Our implementation keeps that idea and adds stacking.
            </div>
            <div class="soft-band">
                <strong>Methodology:</strong> We merged train, stores, and features data, created calendar, holiday, festival, monsoon, lag, and rolling features, then validated on future 2012 records.
            </div>
            <div class="soft-band">
                <strong>Hybrid contribution:</strong> The Ridge meta-learner takes CatBoost and LightGBM predictions as inputs and learns how to combine them, which is more adaptive than fixed MAE weights.
            </div>
            <div class="soft-band">
                <strong>Deployment:</strong> The trained artifact is saved as <code>source/hybrid_stacking_model.joblib</code> and loaded in Streamlit for live prediction and presentation.
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_report_alignment(presentation_mode: bool) -> None:
    hero(
        "Report Alignment",
        "The Implementation Extends the Report, It Does Not Conflict With It",
        "This page gives a clean explanation of how the report's proposed methodology maps to the current Streamlit project.",
    )

    st.markdown("### Core Alignment")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            """
            <div class="info-card">
                <strong>What the report proposed</strong><br><br>
                Retail sales forecasting using gradient boosting models such as XGBoost, LightGBM, and CatBoost, followed by MAE-weighted voting.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            """
            <div class="info-card">
                <strong>What this project implements</strong><br><br>
                CatBoost and LightGBM base learners, MAE-weighted voting baseline, and a Ridge Regression stacking meta-learner.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Extension Logic")
    st.markdown(
        """
        The report's ensemble uses fixed MAE-based weights. That means the final blend is calculated from validation error,
        but the ensemble itself does not learn a new model. This project keeps that baseline and adds a stacking hybrid
        model, where Ridge Regression learns the relationship between base model predictions and actual sales.
        """
    )

    review_box(
        "The report used MAE-weighted voting as the ensemble idea. We implemented that baseline and extended it with stacking, where Ridge Regression learns how much to trust CatBoost and LightGBM predictions."
    )

    st.markdown("### Report vs Implementation")
    report_comparison_table()

    st.markdown("### Defense Notes")
    st.markdown(
        """
        - XGBoost appears in the report, while this implementation focuses on CatBoost and LightGBM due to time and deployment simplicity.
        - The ensemble concept is still matched because MAE-weighted voting is implemented.
        - The hybrid model is an improvement because the final combination is learned, not only assigned from fixed error weights.
        - Deployment is added value beyond the report because the model is accessible through Streamlit.
        """
    )


def page_training_pipeline(presentation_mode: bool) -> None:
    hero(
        "Training Pipeline Flowchart",
        "Interactive Step-by-Step Model Building Pipeline",
        "Expand each stage to explain what was done, why it matters, and how it connects back to the research report.",
    )

    render_flowchart(expand_all=presentation_mode, control_key="training_pipeline_flowchart")

    st.markdown("### Numbered Training Steps")
    render_training_steps()

    review_box(
        "This pipeline proves the model was built like a forecasting system: historical data is transformed into model-ready features, future data is held out for validation, and the final trained artifact is deployed."
    )


def page_results_dashboard(presentation_mode: bool) -> None:
    hero(
        "Results Dashboard",
        "Model Performance and Visual Evidence",
        "Compare CatBoost, LightGBM, MAE-weighted ensemble, and the stacking hybrid model using the saved project metrics and generated plots.",
    )

    metrics = metrics_dict()
    df = comparison_df()

    if metrics:
        best_mae_model = min(metrics, key=lambda key: metrics[key].get("MAE", float("inf")))
        best_r2_model = max(metrics, key=lambda key: metrics[key].get("R2", float("-inf")))
        hybrid = metrics.get("Stacking_Hybrid_Model", {})

        cols = st.columns(4)
        with cols[0]:
            metric_card("Best MAE", best_mae_model.replace("_", " "), f"{metrics[best_mae_model]['MAE']:.2f}")
        with cols[1]:
            metric_card("Best R2", best_r2_model.replace("_", " "), f"{metrics[best_r2_model]['R2']:.4f}")
        with cols[2]:
            metric_card("Hybrid MAE", f"{hybrid.get('MAE', 0):.2f}", "Stacking model")
        with cols[3]:
            metric_card("Hybrid RMSE", f"{hybrid.get('RMSE', 0):.2f}", "Root mean squared error")
    else:
        st.warning(f"Missing or unreadable metrics file: {source_label('metrics.json')}")

    st.markdown("### Model Comparison")
    if not df.empty:
        st.dataframe(df, width="stretch", hide_index=True)

        st.markdown("### Metric Definitions")
        st.caption("Expand these notes while presenting the result table.")
        render_detail_expanders(
            metric_definition_details(metrics),
            control_key="metric_definition_notes",
            default_expanded=presentation_mode,
        )

        metric_cols = [col for col in ["MAE", "RMSE", "R2"] if col in df.columns]
        if metric_cols:
            st.bar_chart(df.set_index("Model")[metric_cols])
    else:
        st.warning(f"Missing or unreadable comparison CSV: {source_label('model_comparison_metrics.csv')}")

    st.markdown("### Validation Plots")
    plot_notes_expanded = expand_collapse_controls(
        "validation_plot_notes",
        default_expanded=presentation_mode,
        expand_label="Expand all",
        collapse_label="Collapse all",
    )
    plot_tabs = st.tabs(["Actual vs Predicted", "Error Distribution", "Sales Over Time"])
    with plot_tabs[0]:
        safe_image("actual_vs_predicted_hybrid.png", "Stacking Hybrid Model: Actual vs Predicted")
        render_plot_explanation("Actual vs Predicted", plot_notes_expanded)
    with plot_tabs[1]:
        safe_image("error_distribution_hybrid.png", "Stacking Hybrid Model: Error Distribution")
        render_plot_explanation("Error Distribution", plot_notes_expanded)
    with plot_tabs[2]:
        safe_image("sales_over_time_hybrid.png", "Stacking Hybrid Model: Sales Over Time")
        render_plot_explanation("Sales Over Time", plot_notes_expanded)

    review_box(
        "These results show that the gradient boosting models perform strongly, the MAE-weighted ensemble baseline is available, and the stacking hybrid model is evaluated with the same report metrics."
    )


def page_hybrid_explanation(presentation_mode: bool) -> None:
    hero(
        "Hybrid Model Explanation",
        "From Fixed Voting to Learned Stacking",
        "This page explains the model architecture in reviewer-friendly language.",
    )

    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.markdown(
            """
            <div class="info-card">
                <strong>Base Learner 1</strong><br><br>
                CatBoost learns from categorical store, department, type, festival, and numeric sales-history features.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            """
            <div class="info-card">
                <strong>Base Learner 2</strong><br><br>
                LightGBM learns a fast gradient boosting model from encoded categorical and numeric features.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            """
            <div class="info-card">
                <strong>Meta Learner</strong><br><br>
                Ridge Regression takes the two base predictions and learns the final hybrid forecast.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Architecture")
    st.code(
        """
Input features
    -> CatBoost prediction
    -> LightGBM prediction
        -> Ridge Regression meta-learner
            -> Final weekly sales forecast
        """.strip(),
        language="text",
    )

    st.markdown("### Model Definitions and Hybrid Details")
    st.caption("Use these expanders when explaining what each model is and why it is used here.")
    render_detail_expanders(
        MODEL_EXPLANATION_DETAILS,
        control_key="model_explanation_notes",
        default_expanded=presentation_mode,
    )

    meta = meta_dict()
    if meta:
        st.markdown("### Learned Meta-Learner Coefficients")
        st.markdown(
            """
            <div class="soft-band hover-tip" data-tip="These values belong to the Ridge Regression meta-learner. They explain how the final stacking model combines the two base model predictions.">
                <strong>What this means:</strong> the hybrid model first gets one prediction from CatBoost and one prediction from LightGBM.
                Ridge Regression then applies these learned coefficients and the intercept to produce the final weekly sales forecast.
                Hover over each card for the presentation definition.
            </div>
            """,
            unsafe_allow_html=True,
        )
        coef_cols = st.columns(3)
        with coef_cols[0]:
            coefficient_card(
                "CatBoost coefficient",
                f"{meta.get('CatBoost_prediction_coefficient', 0):.4f}",
                "Ridge multiplier for CatBoost prediction",
                "This is the learned multiplier applied to the CatBoost base prediction inside the Ridge stacking model. A negative value means Ridge uses CatBoost as a correction signal after considering LightGBM, not that CatBoost is useless.",
            )
        with coef_cols[1]:
            coefficient_card(
                "LightGBM coefficient",
                f"{meta.get('LightGBM_prediction_coefficient', 0):.4f}",
                "Ridge multiplier for LightGBM prediction",
                "This is the learned multiplier applied to the LightGBM base prediction. Since LightGBM has the best MAE in the results, Ridge gives it the strongest positive influence in the final hybrid forecast.",
            )
        with coef_cols[2]:
            coefficient_card(
                "Intercept",
                f"{meta.get('Intercept', 0):.4f}",
                "Final Ridge adjustment term",
                "The intercept is a small constant adjustment added after multiplying the CatBoost and LightGBM predictions by their coefficients.",
            )
    else:
        st.warning(f"Missing or unreadable coefficient file: {source_label('meta_coefficients.json')}")

    st.markdown("### MAE Weighted Ensemble vs Stacking")
    compare_cols = st.columns(2)
    with compare_cols[0]:
        st.markdown(
            """
            <div class="info-card">
                <strong>MAE-weighted voting</strong><br><br>
                Uses fixed inverse-error weights. It is simple, explainable, and directly matches the report baseline.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with compare_cols[1]:
        st.markdown(
            """
            <div class="info-card">
                <strong>Stacking hybrid model</strong><br><br>
                Trains Ridge Regression on base model predictions, so the final blend is learned from data.
            </div>
            """,
            unsafe_allow_html=True,
        )

    review_box(
        "We extended the report's ensemble method by training a stacking-based hybrid model where Ridge Regression learns how to combine CatBoost and LightGBM predictions."
    )


def prediction_form_defaults() -> dict[str, Any]:
    return {
        "Store": 1,
        "Dept": 1,
        "Type": "A",
        "Size": 151315,
        "IsHoliday": 0,
        "Year": 2012,
        "Month": 6,
        "WeekOfYear": 24,
        "IsMonthEnd": 0,
        "Indian_Festival": "None",
        "Is_Indian_Festival": 0,
        "Is_Monsoon": 1,
        "Sales_Lag_1": 22000.0,
        "Sales_Lag_4": 21500.0,
        "Sales_Lag_12": 20800.0,
        "Sales_RollMean_4": 21850.0,
        "Sales_RollStd_12": 1200.0,
    }


def make_prediction(artifact: dict[str, Any], values: dict[str, Any]) -> float:
    feature_cols = artifact.get("feature_cols", list(values.keys()))
    cat_features = artifact.get("cat_features", ["Store", "Dept", "Type", "Indian_Festival"])

    row = pd.DataFrame([{col: values.get(col) for col in feature_cols}])

    for col in cat_features:
        if col in row:
            row[col] = row[col].astype(str)

    cat_pred = artifact["catboost_model"].predict(row)

    row_lgb = row.copy()
    row_lgb[cat_features] = artifact["lightgbm_encoder"].transform(row_lgb[cat_features].astype(str))
    lgb_pred = artifact["lightgbm_model"].predict(row_lgb)

    meta_input = np.column_stack([cat_pred, lgb_pred])
    final_pred = artifact["meta_model"].predict(meta_input)
    return float(final_pred[0])


def page_prediction_demo(presentation_mode: bool) -> None:
    hero(
        "Prediction Demo",
        "Live Forecast From the Saved Hybrid Model",
        "Enter one Store-Dept-week feature set and generate a weekly sales prediction using the saved joblib artifact.",
    )

    st.markdown(
        """
        <div class="demo-grid">
            <div class="demo-card hover-tip" data-tip="These fields describe which store, department, week, and seasonal context the model should forecast.">
                <div class="demo-label">Input Layer</div>
                <div class="demo-title">Store-week feature row</div>
                <div class="demo-text">The form builds one model-ready row with identity, calendar, event, and sales-memory features.</div>
            </div>
            <div class="demo-card hover-tip" data-tip="Lag and rolling fields give the forecast recent sales memory, which is essential for time-series style prediction.">
                <div class="demo-label">Forecast Memory</div>
                <div class="demo-title">Lag and rolling signals</div>
                <div class="demo-text">Recent sales values and rolling statistics help the model understand demand level and volatility.</div>
            </div>
            <div class="demo-card hover-tip" data-tip="CatBoost and LightGBM predict first. Ridge Regression then combines those two predictions into the final forecast.">
                <div class="demo-label">Hybrid Output</div>
                <div class="demo-title">CatBoost + LightGBM + Ridge</div>
                <div class="demo-text">The saved stacking model converts the feature row into one final weekly sales forecast.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    artifact, load_error = load_model_artifact(str(MODEL_PATH))
    if load_error:
        st.warning(load_error)
        st.info("The rest of the dashboard still works because model loading is isolated from the presentation pages.")
        return

    defaults = prediction_form_defaults()
    with st.container(border=True):
        st.markdown(
            """
            <div class="form-section-title">
                Store and Department Context
                <span>These fields identify where the forecast is being made.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        top_cols = st.columns(4)
        with top_cols[0]:
            defaults["Store"] = st.number_input(
                "Store",
                min_value=1,
                max_value=45,
                value=defaults["Store"],
                step=1,
                help=PREDICTION_FIELD_HELP["Store"],
            )
        with top_cols[1]:
            defaults["Dept"] = st.number_input(
                "Dept",
                min_value=1,
                max_value=99,
                value=defaults["Dept"],
                step=1,
                help=PREDICTION_FIELD_HELP["Dept"],
            )
        with top_cols[2]:
            defaults["Type"] = st.selectbox(
                "Store Type",
                ["A", "B", "C"],
                index=0,
                help=PREDICTION_FIELD_HELP["Type"],
            )
        with top_cols[3]:
            defaults["Size"] = st.number_input(
                "Store Size",
                min_value=1,
                value=defaults["Size"],
                step=1000,
                help=PREDICTION_FIELD_HELP["Size"],
            )

        st.markdown(
            """
            <div class="form-section-title">
                Calendar and Event Signals
                <span>These fields tell the model when the forecast happens and whether special periods apply.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        date_cols = st.columns(5)
        with date_cols[0]:
            defaults["Year"] = st.number_input(
                "Year",
                min_value=2010,
                max_value=2030,
                value=defaults["Year"],
                step=1,
                help=PREDICTION_FIELD_HELP["Year"],
            )
        with date_cols[1]:
            defaults["Month"] = st.number_input(
                "Month",
                min_value=1,
                max_value=12,
                value=defaults["Month"],
                step=1,
                help=PREDICTION_FIELD_HELP["Month"],
            )
        with date_cols[2]:
            defaults["WeekOfYear"] = st.number_input(
                "Week of Year",
                min_value=1,
                max_value=53,
                value=defaults["WeekOfYear"],
                step=1,
                help=PREDICTION_FIELD_HELP["WeekOfYear"],
            )
        with date_cols[3]:
            defaults["IsHoliday"] = int(
                st.checkbox(
                    "Holiday Week",
                    value=bool(defaults["IsHoliday"]),
                    help=PREDICTION_FIELD_HELP["IsHoliday"],
                )
            )
        with date_cols[4]:
            defaults["IsMonthEnd"] = int(
                st.checkbox(
                    "Month End",
                    value=bool(defaults["IsMonthEnd"]),
                    help=PREDICTION_FIELD_HELP["IsMonthEnd"],
                )
            )

        context_cols = st.columns(3)
        with context_cols[0]:
            defaults["Indian_Festival"] = st.selectbox(
                "Indian Festival",
                ["None", "Holi", "Eid", "Diwali", "Christmas"],
                index=0,
                help=PREDICTION_FIELD_HELP["Indian_Festival"],
            )
        with context_cols[1]:
            defaults["Is_Indian_Festival"] = int(defaults["Indian_Festival"] != "None")
            st.markdown(
                f"""
                <div class="derived-card hover-tip" data-tip="{PREDICTION_FIELD_HELP['Is_Indian_Festival']}">
                    <div class="demo-label">Derived Field</div>
                    <div class="demo-title">Festival Flag</div>
                    <div class="derived-value">{defaults["Is_Indian_Festival"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with context_cols[2]:
            defaults["Is_Monsoon"] = int(
                st.checkbox(
                    "Monsoon Season",
                    value=bool(defaults["Is_Monsoon"]),
                    help=PREDICTION_FIELD_HELP["Is_Monsoon"],
                )
            )

        st.markdown(
            """
            <div class="form-section-title">
                Sales Memory Features
                <span>These values give the model historical demand context for the same store and department.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        lag_cols = st.columns(5)
        with lag_cols[0]:
            defaults["Sales_Lag_1"] = st.number_input(
                "Sales Lag 1",
                min_value=0.0,
                value=defaults["Sales_Lag_1"],
                step=500.0,
                format="%.2f",
                help=PREDICTION_FIELD_HELP["Sales_Lag_1"],
            )
        with lag_cols[1]:
            defaults["Sales_Lag_4"] = st.number_input(
                "Sales Lag 4",
                min_value=0.0,
                value=defaults["Sales_Lag_4"],
                step=500.0,
                format="%.2f",
                help=PREDICTION_FIELD_HELP["Sales_Lag_4"],
            )
        with lag_cols[2]:
            defaults["Sales_Lag_12"] = st.number_input(
                "Sales Lag 12",
                min_value=0.0,
                value=defaults["Sales_Lag_12"],
                step=500.0,
                format="%.2f",
                help=PREDICTION_FIELD_HELP["Sales_Lag_12"],
            )
        with lag_cols[3]:
            defaults["Sales_RollMean_4"] = st.number_input(
                "Rolling Mean 4",
                min_value=0.0,
                value=defaults["Sales_RollMean_4"],
                step=500.0,
                format="%.2f",
                help=PREDICTION_FIELD_HELP["Sales_RollMean_4"],
            )
        with lag_cols[4]:
            defaults["Sales_RollStd_12"] = st.number_input(
                "Rolling Std 12",
                min_value=0.0,
                value=defaults["Sales_RollStd_12"],
                step=100.0,
                format="%.2f",
                help=PREDICTION_FIELD_HELP["Sales_RollStd_12"],
            )

        submitted = st.button("Predict Weekly Sales", type="primary", use_container_width=True)

    if submitted:
        try:
            prediction = make_prediction(artifact, defaults)
            st.markdown(
                f"""
                <div class="prediction-result">
                    <small>Predicted Weekly Sales</small>
                    <strong>{prediction:,.2f}</strong>
                    <div class="demo-text">Generated by CatBoost and LightGBM base predictions blended through the Ridge stacking layer.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.warning(f"Prediction failed, but the app stayed running. Details: {exc}")

    review_box(
        "The final trained model is saved as a joblib artifact and loaded by Streamlit, so the project moves beyond a notebook into an interactive deployment."
    )


def page_about_viva(presentation_mode: bool) -> None:
    hero(
        "About / Viva",
        "Concise Answers for Project Review",
        "Use these points to answer common questions about dataset choice, methodology, model design, and limitations.",
    )

    st.markdown(
        """
        <div class="viva-grid">
            <div class="viva-card hover-tip" data-tip="Use this as your first sentence if the reviewer asks what the project is about.">
                <div class="viva-label">Project Idea</div>
                <div class="viva-title">Retail sales forecasting</div>
                <div class="viva-text">Predict weekly sales using store, department, calendar, event, and sales-history features.</div>
            </div>
            <div class="viva-card hover-tip" data-tip="This is the model story: two strong base learners and one learned final combiner.">
                <div class="viva-label">Model Design</div>
                <div class="viva-title">CatBoost + LightGBM + Ridge</div>
                <div class="viva-text">Gradient boosting models produce base predictions, then Ridge Regression learns the final hybrid blend.</div>
            </div>
            <div class="viva-card hover-tip" data-tip="This is the review contribution beyond training a notebook model.">
                <div class="viva-label">Deployment</div>
                <div class="viva-title">Streamlit defense dashboard</div>
                <div class="viva-text">The app explains methodology, metrics, plots, limitations, and live prediction from the saved model artifact.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="soft-band">
            <strong>Opening pitch:</strong> This project builds an end-to-end retail sales forecasting system using the Walmart dataset.
            It follows the report's gradient boosting and MAE-weighted ensemble idea, then extends it with a stacking hybrid model and deploys the result in Streamlit.
        </div>
        """,
        unsafe_allow_html=True,
    )

    qa_items = [
        (
            "Why use a time-based split?",
            [
                "Forecasting should imitate reality: learn from past weeks and predict future weeks.",
                "A random split could leak future patterns into training and make the results look too good.",
                "Using future 2012 records for validation makes MAE, RMSE, and R2 more defensible.",
            ],
        ),
        (
            "Why CatBoost?",
            [
                "CatBoost is strong for tabular datasets with categorical columns.",
                "It can use Store, Dept, Type, and Indian_Festival along with numerical sales-history features.",
                "It gives one diverse base prediction for the final hybrid model.",
            ],
        ),
        (
            "Why LightGBM?",
            [
                "LightGBM is fast and accurate for large structured datasets.",
                "It learns nonlinear interactions between lag features, rolling features, calendar fields, and store information.",
                "In the saved results, it has the best MAE, so it is a strong base learner.",
            ],
        ),
        (
            "Why add Indian festival and monsoon features to a Walmart dataset?",
            [
                "They demonstrate context-aware feature engineering beyond the original columns.",
                "In a local retail setting, festivals and seasonal periods can change shopping behavior.",
                "For presentation, they show how the same forecasting pipeline can be adapted to regional retail demand.",
            ],
        ),
        (
            "How is stacking different from MAE voting?",
            [
                "MAE voting uses fixed weights based on validation error.",
                "Stacking trains a Ridge Regression meta-learner on CatBoost and LightGBM predictions.",
                "So stacking learns the final combination, while MAE voting only applies a rule-based weighted average.",
            ],
        ),
        (
            "What is one limitation?",
            [
                "A production-grade stacking model should train the meta-learner using out-of-fold predictions.",
                "For time-series forecasting, TimeSeriesSplit would make this more rigorous.",
                "This is a good future-work point because it improves stacking reliability without changing the project idea.",
            ],
        ),
    ]

    st.markdown("### Review Answer Bank")
    qa_expanded = expand_collapse_controls(
        "viva_answer_bank",
        default_expanded=presentation_mode,
        expand_label="Expand all",
        collapse_label="Collapse all",
    )
    for question, answer in qa_items:
        with st.expander(question, expanded=qa_expanded):
            render_bullet_section("Answer points", answer)

    st.markdown("### Files Used by the UI")
    artifact_items = [
        ("Model artifact", "hybrid_stacking_model.joblib", "Saved CatBoost, LightGBM, encoder, Ridge meta-model, and feature metadata."),
        ("Metrics", "metrics.json", "MAE, MSE, RMSE, and R2 for the trained models."),
        ("Meta coefficients", "meta_coefficients.json", "Learned Ridge coefficients used to explain the stacking layer."),
        ("Comparison table", "model_comparison_metrics.csv", "Tabular model comparison shown in the Results Dashboard."),
        ("Actual vs predicted", "actual_vs_predicted_hybrid.png", "Validation scatter plot showing predicted sales against actual sales."),
        ("Error distribution", "error_distribution_hybrid.png", "Histogram showing where prediction errors are concentrated."),
        ("Sales over time", "sales_over_time_hybrid.png", "Time plot comparing actual and predicted weekly sales movement."),
    ]
    artifact_cards = []
    for title, filename, description in artifact_items:
        exists = (SOURCE_DIR / filename).exists()
        status_class = "artifact-ok" if exists else "artifact-missing"
        status_text = "Ready" if exists else "Missing"
        artifact_cards.append(
            f"""
            <div class="artifact-card hover-tip" data-tip="{description}">
                <span class="artifact-status {status_class}">{status_text}</span>
                <div>
                    <div class="artifact-title">{title}</div>
                    <div class="artifact-path">source/{filename}</div>
                    <div class="artifact-text">{description}</div>
                </div>
            </div>
            """
        )

    st.html('<div class="artifact-grid">' + "".join(artifact_cards) + "</div>")

    review_box(
        "My contribution is the full bridge from research methodology to implementation: feature engineering, gradient boosting models, weighted ensemble baseline, stacking hybrid model, evaluation, and Streamlit deployment."
    )


def render_sidebar() -> tuple[str, bool]:
    st.sidebar.title("Forecasting UI")
    st.sidebar.caption("Mini project defense dashboard")
    page = st.sidebar.radio("Navigate", PAGE_OPTIONS, index=0)
    presentation_mode = st.sidebar.checkbox(
        "Presentation Mode",
        value=False,
        help="Increases text emphasis and opens walkthrough details for review explanation.",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Artifact base path**")
    st.sidebar.code("source/")
    return page, presentation_mode


def main() -> None:
    st.set_page_config(
        page_title="Hybrid Sales Forecasting UI",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    page, presentation_mode = render_sidebar()
    apply_css(presentation_mode)

    if page == "Home":
        page_home(presentation_mode)
    elif page == "One-Slide Summary":
        page_one_slide_summary(presentation_mode)
    elif page == "Project Walkthrough":
        page_project_walkthrough(presentation_mode)
    elif page == "Report Alignment":
        page_report_alignment(presentation_mode)
    elif page == "Training Pipeline Flowchart":
        page_training_pipeline(presentation_mode)
    elif page == "Results Dashboard":
        page_results_dashboard(presentation_mode)
    elif page == "Hybrid Model Explanation":
        page_hybrid_explanation(presentation_mode)
    elif page == "Prediction Demo":
        page_prediction_demo(presentation_mode)
    elif page == "About / Viva":
        page_about_viva(presentation_mode)


if __name__ == "__main__":
    main()
