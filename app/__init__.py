import os

from flask import Flask, redirect, url_for

"""Create and configure an instance of the Flask application."""

app = Flask(__name__, instance_relative_config=True)
app.config.from_mapping(
    # a default secret that should be overridden by instance config
    SECRET_KEY="dev",
)

# load the instance config, if it exists, when not testing
app.config.from_pyfile("config.py", silent=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5432, debug=True)

# ensure the instance folder exists
try:
    os.makedirs(app.instance_path)
except OSError:
    pass

from . import query, annotations, videos
app.register_blueprint(query.bp)
app.register_blueprint(annotations.bp)
app.register_blueprint(videos.bp)


# Redirect the root URL to "/query"
@app.route("/")
def root():
    return redirect(url_for("query.query"))
