import os

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

bp = Blueprint('query', __name__)


@bp.route("/", methods=("GET", "POST"))
def query():
    """Query for a video."""
    if request.method == "POST":
        query_input = request.form["query"]
        selected_field = request.form.get('field')
        error = None

        if not query_input:
            error = "Query is required."

        if error is not None:
            flash(error)
        else:
            print("Query: " + query_input)
            print("Field: " + str(selected_field))

            # TODO: Query opensearch for videos matching query
            session['expressions'] = ["exp9", "exp10"]
            session['videos'] = ["9", "10"]

            return redirect(url_for("query.results"))

    return render_template("query/query.html")


@bp.route("/results", methods=("GET", "POST"))
def results():
    """Display results of query."""
    expressions = session.get('expressions', [])

    frames = {}
    frames_path = os.path.join(os.getcwd(), 'app', 'static', 'videofiles', 'frames')

    # Get the facial expression frames
    for exp_id in expressions:
        frames[exp_id] = os.listdir(os.path.join(frames_path, exp_id))

    if request.method == "POST":
        selected_expression = request.form.get("expression")
        error = None

        if not selected_expression:
            error = "Expression is required."

        if error is not None:
            flash(error)
        else:
            print("Selected expression: " + selected_expression)

            return redirect(url_for("query.play_expression", exp_id=selected_expression))

    return render_template("query/results.html", expressions=expressions, frames=frames)


@bp.route("/results/<exp_id>", methods=("GET", "POST"))
def play_expression(exp_id):
    """ Display the video clip of the selected expression. """

    videos = session.get('videos', [])

    # TODO: Get the video clip of the selected expression using the timestamps
    video = videos[0]

    return render_template("query/play_expression.html", exp_id=exp_id, video=video)
