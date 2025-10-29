from flask import Blueprint, render_template
import json
import os

task4_bp = Blueprint('task4', __name__, template_folder='templates', static_folder='static')

@task4_bp.route('/')
def home():
    # Load metrics from JSON file
    metrics_path = os.path.join(os.path.dirname(__file__), 'metrics.json')
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return render_template('index4.html', metrics=metrics)
