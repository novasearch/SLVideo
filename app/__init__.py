import os

from flask import Flask

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

from . import query

app.register_blueprint(query.bp)

app.add_url_rule("/", endpoint="query")
