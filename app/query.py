import os

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

FRAMES_DIR = "app/static/videofiles/frames"

bp = Blueprint('query', __name__)


@bp.route("/", methods=("GET", "POST"))
def query():
    """Query for a video."""
    if request.method == "POST":
        query_input = request.form["query"]
        selected_field = int(request.form.get('field'))
        error = None

        if not query_input:
            error = "Query is required."

        if error is not None:
            flash(error)
        else:
            print("Query: " + query_input)
            print("Field: " + str(selected_field))

            # TODO: Query opensearch for videos matching query
            if selected_field == 1:  # Frames Embeddings
                flash("Not Implemented")
            elif selected_field == 2:  # Average Frames Embeddings
                flash("Not Implemented")
            elif selected_field == 3:  # Best Frame Embedding
                flash("Not Implemented")
            elif selected_field == 4:  # True Expression
                session['expressions'] = query_true_expression(query_input)

            return redirect(url_for("query.results"))

    return render_template("query/query.html")


@bp.route("/results", methods=("GET", "POST"))
def results():
    """Display results of query."""
    expressions = session.get('expressions', [])

    frames = {}

    # Not all frames of each expression are going to be displayed
    frames_to_display = {}
    num_frames_to_display = 8

    # Get the facial expression frames
    for exp_path in expressions:
        frames[exp_path] = os.listdir(os.path.join(FRAMES_DIR, exp_path))

    # Display only #num_frames_to_display frames of each expression
    print("AAAAAAAAA")
    for expression, all_frames in frames.items():
        print("VBBBBBBBB")

        # Calculate the step size
        if len(all_frames) <= num_frames_to_display:
            step_size = 1
        else:
            # Calculate the step size
            step_size = (len(all_frames) - 1) // num_frames_to_display + 1

        # Select frames to display
        frames_to_display[expression] = all_frames[::step_size]

        # Ensure that only num_frames_to_display frames are selected
        frames_to_display[expression] = frames_to_display[expression][:num_frames_to_display]

        print("Step_size:", step_size)
        print("frames_to_display:", frames_to_display)

    if request.method == "POST":
        selected_expression = request.form.get("expression")
        error = None

        if not selected_expression:
            error = "Expression is required."

        if error is not None:
            flash(error)
        else:

            # Replace '/' with '_', so that the expression can be used as a URL parameter
            modified_expression = selected_expression.replace('/', '_')

            return redirect(url_for("query.play_expression", exp_id=modified_expression))

    return render_template("query/results.html", expressions=expressions, frames=frames_to_display)


@bp.route("/results/<exp_id>", methods=("GET", "POST"))
def play_expression(exp_id):
    """ Display the video clip of the selected expression. """

    # The expression is in the format: video_expression_timestamp
    video, expression, timestamp = exp_id.split('_')

    # Convert the timestamp to seconds
    timestamp_seconds = int(timestamp) // 1000

    return render_template("query/play_expression.html", expression=expression, video=video, timestamp=timestamp_seconds)


def query_true_expression(query_input):
    """ Get the results of the query using the ground truth """
    expressions = []

    for video_expressions in os.listdir(FRAMES_DIR):
        for expression in os.listdir(os.path.join(FRAMES_DIR, video_expressions)):
            if query_input.lower() in expression.lower():
                #expressions.append(os.path.join(video_expressions, expression))
                expressions.append(video_expressions + "/" + expression)

    print("Expressions: " + str(expressions))
    return expressions
