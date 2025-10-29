from flask import Flask, render_template
from TASK_1.app import task1_bp
from TASK_2.app import task2_bp
from TASK_3.app import task3_bp
from TASK_4.app import task4_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(task1_bp, url_prefix='/task1')
app.register_blueprint(task2_bp, url_prefix='/task2')
app.register_blueprint(task3_bp, url_prefix='/task3')
app.register_blueprint(task4_bp, url_prefix='/task4')

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
