import os

from flask import Blueprint, render_template

from app.constants import VIDEO_PATH, FRAMES_PATH, PHRASES_ID

bp = Blueprint('videos', __name__)


@bp.route('/videos')
def list_videos():
    """ Shows all the videos in the database. """

    videos = {}

    for video in os.listdir(VIDEO_PATH):
        video_name = video.split('.')[0]
        frames_path = os.path.join(FRAMES_PATH, PHRASES_ID, video_name)
        first_annotation_path = os.listdir(frames_path)[0]
        thumbnail = os.listdir(os.path.join(frames_path, first_annotation_path))[0]

        videos[video_name] = {}
        videos[video_name]['path'] = os.path.join(VIDEO_PATH, video)
        videos[video_name]['thumbnail'] = os.path.join(PHRASES_ID, video_name, first_annotation_path, thumbnail)

    return render_template("videos_list/videos_list.html", videos=videos)
