from flask import Blueprint, render_template

task4_bp = Blueprint('task4', __name__, template_folder='templates', static_folder='static')

# -----------------------------------
# Home Route
# -----------------------------------
@task4_bp.route("/")
def home():
    # Directly render index.html that shows plots & map
    return render_template("index4.html")
