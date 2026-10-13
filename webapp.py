"""Flask dashboard for the Wisconsin diagnostic classification case study."""

from functools import lru_cache

import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request
from markupsafe import Markup
from sklearn.metrics import confusion_matrix, roc_curve

from model import calculate_metrics, load_dataset, train_and_evaluate


app = Flask(__name__)


@lru_cache(maxsize=1)
def get_results():
    return train_and_evaluate()


def chart_html(figure, include_plotlyjs=False):
    return Markup(
        figure.to_html(
            full_html=False,
            include_plotlyjs="cdn" if include_plotlyjs else False,
            config={"displayModeBar": False, "responsive": True},
        )
    )


@app.get("/")
def dashboard():
    features, target = load_dataset()
    results = get_results()
    try:
        threshold = float(request.args.get("threshold", "0.50"))
    except ValueError:
        threshold = 0.5
    threshold = min(0.95, max(0.05, threshold))

    available_indices = results["x_test"].index.tolist()
    try:
        sample_index = int(request.args.get("sample", available_indices[0]))
    except ValueError:
        sample_index = available_indices[0]
    if sample_index not in available_indices:
        sample_index = available_indices[0]

    metrics = calculate_metrics(results["y_test"], results["probabilities"], threshold)
    malignant_count = int(target.sum())
    benign_count = int((target == 0).sum())

    class_figure = px.bar(
        x=["Benign", "Malignant"],
        y=[benign_count, malignant_count],
        color=["Benign", "Malignant"],
        color_discrete_map={"Benign": "#168c83", "Malignant": "#d45c47"},
        labels={"x": "Diagnosis", "y": "Samples", "color": "Diagnosis"},
        title="Class balance",
    )
    class_figure.update_layout(
        showlegend=False, margin={"t": 48, "b": 10, "l": 10, "r": 10}
    )

    scatter_data = features[["mean radius", "mean texture"]].copy()
    scatter_data["Diagnosis"] = target.map({0: "Benign", 1: "Malignant"})
    scatter_figure = px.scatter(
        scatter_data,
        x="mean radius",
        y="mean texture",
        color="Diagnosis",
        color_discrete_map={"Benign": "#168c83", "Malignant": "#d45c47"},
        opacity=0.75,
        title="Radius and texture measurements",
    )
    scatter_figure.update_layout(margin={"t": 48, "b": 10, "l": 10, "r": 10})

    matrix = confusion_matrix(
        results["y_test"],
        (results["probabilities"] >= threshold).astype(int),
        labels=[0, 1],
    )
    matrix_figure = px.imshow(
        matrix,
        text_auto=True,
        x=["Predicted benign", "Predicted malignant"],
        y=["Actual benign", "Actual malignant"],
        color_continuous_scale="Teal",
        title=f"Confusion matrix at threshold {threshold:.2f}",
    )
    matrix_figure.update_layout(margin={"t": 48, "b": 10, "l": 10, "r": 10})

    false_positive_rate, true_positive_rate, _ = roc_curve(
        results["y_test"], results["probabilities"]
    )
    roc_figure = go.Figure()
    roc_figure.add_trace(
        go.Scatter(x=false_positive_rate, y=true_positive_rate, mode="lines", name="Model")
    )
    roc_figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line={"dash": "dash"},
        )
    )
    roc_figure.update_layout(
        title="ROC curve on held-out test set",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        yaxis_range=[0, 1.02],
        margin={"t": 48, "b": 10, "l": 10, "r": 10},
    )

    top_features = results["importance"].head(12).sort_values("Importance")
    importance_figure = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        error_x="SD",
        color="Importance",
        color_continuous_scale="Teal",
        title="Top permutation importances",
    )
    importance_figure.update_layout(margin={"t": 48, "b": 10, "l": 10, "r": 10})

    sample = results["x_test"].loc[[sample_index]]
    probability = float(results["model"].predict_proba(sample)[0, 1])
    actual_class = int(results["y_test"].loc[sample_index])
    leaderboard = results["leaderboard"].to_dict(orient="records")

    return render_template(
        "index.html",
        total=len(target),
        feature_count=features.shape[1],
        malignant_count=malignant_count,
        malignant_rate=malignant_count / len(target),
        model_name=results["model_name"],
        leaderboard=leaderboard,
        metrics=metrics,
        threshold=threshold,
        sample_index=sample_index,
        sample_options=available_indices,
        probability=probability,
        prediction="Malignant" if probability >= 0.5 else "Benign",
        actual="Malignant" if actual_class else "Benign",
        sample_measurements=sample.iloc[0].items(),
        class_chart=chart_html(class_figure, include_plotlyjs=True),
        scatter_chart=chart_html(scatter_figure),
        matrix_chart=chart_html(matrix_figure),
        roc_chart=chart_html(roc_figure),
        importance_chart=chart_html(importance_figure),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)