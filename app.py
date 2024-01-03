from flask import Flask, render_template, send_from_directory, g
import ffmpeg

from eaf_parser.eaf_parser import get_time_slot_of_phrase, get_first_time_slot_value

app = Flask(__name__)

start_time = int(get_first_time_slot_value())

VIDEO_FILE = '9.mp4'


@app.route('/')
def home():
    video = ffmpeg.input('videofiles/' + VIDEO_FILE)
    ffmpeg.output(video, 'videofiles/output.mp4').overwrite_output().run()

    return render_template('app.html', video_filename='output.mp4')


@app.route('/videofiles/<filename>')
def video(filename):
    return send_from_directory('videofiles', filename)

@app.route('/search_keyword/<keyword>', methods=['POST'])
def search_keyword(keyword):
    timestamp = int(get_time_slot_of_phrase(keyword))

    if timestamp is None:
        return render_template('app.html', video_filename=VIDEO_FILE)

    timestamp = (timestamp - start_time) / 1000

    video = ffmpeg.input('videofiles/' + VIDEO_FILE)
    moved_video = video.filter('setpts', 'PTS+' + str(timestamp) + '/TB')

    ffmpeg.output(moved_video, 'videofiles/output.mp4').overwrite_output().run()

    return render_template('app.html', video_filename=VIDEO_FILE)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
