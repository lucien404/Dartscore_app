import os
import json
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from scoring import main
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = './static/img'
RESULT_FOLDER = './static/result'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/img/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route('/', methods=['GET', 'POST'])
def feibiao():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filename_path)
            score = main(filename_path)
            result_dict = {
                'img': filename,
                'score': score
            }
            json_filename = filename.split('.')[0] + '.json'
            json_filename_path = os.path.join(app.config["RESULT_FOLDER"], json_filename)
            with open(json_filename_path, "w") as f:
                json.dump(result_dict, f)
            return redirect(url_for('download_result', name=json_filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    
    <a href="/status" target="_blank">status</a>
    
    <form method=post action="/restart" target="_blank">
      <input type=submit value=restart>
    </form>
    
    <form method=post action="/compare" target="_blank">
      <input type=submit value=compare>
    </form>
    
    <form method=post enctype=multipart/form-data action="/shooting">
      <input type="text" id="playerId" name="playerId">
      <input type=file name=file>
      <input type=submit value=shooting>
    </form>
    '''


@app.route('/result/<name>')
def download_result(name):
    return send_from_directory(app.config["RESULT_FOLDER"], name)


@app.route('/feibiaocv', methods=['POST'])
def feibiaocv():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filename_path)
        score = main(filename_path)
        result_dict = {
            'img': filename,
            'score': score
        }
        json_filename = filename.split('.')[0] + '.json'
        json_filename_path = os.path.join(app.config["RESULT_FOLDER"], json_filename)
        with open(json_filename_path, "w") as f:
            json.dump(result_dict, f)
        return send_from_directory(app.config["RESULT_FOLDER"], json_filename)


@app.route('/status', methods=['GET'])
def status():
    return send_from_directory("./static/", "status.json")


@app.route('/restart', methods=['POST'])
def restart():
    with open("./static/status_restart.json", "r") as f:
        status_dict = json.load(f)
    with open("./static/status.json", "w") as f:
        json.dump(status_dict, f)
    filepath = "./static/pictures"
    shutil.rmtree(filepath)
    os.makedirs(filepath)
    return {}


@app.route('/compare', methods=['POST'])
def compare():
    with open("./static/status.json", "r") as f:
        status_dict = json.load(f)

    win_player_id = []
    max_points = 0
    for player in status_dict["players"]:
        player_points = 0
        for item in player["rounds"]:
            for score in item["points"]:
                player_points = player_points + score
        if player_points > max_points:
            win_player_id = [player["name"]]
        elif player_points == max_points:
            win_player_id.append(player["name"])

    return {
        'success': {
            'playerId': win_player_id
        }
    }


@app.route('/shooting', methods=['POST'])
def shooting():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    player_id = request.form['playerId']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        with open("./static/status.json", "r") as f:
            status_dict = json.load(f)

        selected_index = -1
        for index, player in enumerate(status_dict["players"]):
            if player["name"] == player_id:
                selected_index = index
                break

        if selected_index == -1:
            new_player = {
                'name': player_id,
                'rounds': []
            }
            status_dict["players"].append(new_player)
            selected_index = len(status_dict["players"]) - 1

        filename1 = secure_filename(file.filename)
        filename = str(selected_index) + '.' + filename1.rsplit('.', 1)[1].lower()
        dir_path = "./static/pictures/" + player_id
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filename_path = os.path.join(dir_path, filename)
        file.save(filename_path)

        try:
            score = main(filename_path)
        except Exception as ex:
            return {
                'return': -1,
                'error': {
                    'message': 'The system can’t recognize the dart board, please change the angle or zoom in and shoot again (side view)'
                }
            }

        round_info = {
            "pictureId": filename_path,
            "points": score
        }
        status_dict["players"][selected_index]["rounds"].append(round_info)

        return {
            'return': 0,
            'success': {
                "playerId": player_id,
                "points": score
            }
        }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
