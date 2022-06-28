import flask
import os
import shutil
from flask import request
from Classificator import Classificator
import numpy as np
import uuid


app = flask.Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'tmp'
classifier = Classificator()


@app.route('/', methods=['GET'])
def get_message():
    with open('templates/index.html', 'rt') as f:
        return f.read()


def save_files(request):
    result = {
        "success": False,
        "error": None,
        "data": {},
        "root": ""
    }
    app.config["UPLOAD_FOLDER"] = str(uuid.uuid4())
    for i, f in enumerate(request.files.getlist('file')):
        path_list = f.filename.split('/')
        if len(path_list) != 3:
            continue
            # result["success"] = False
            # result["error"] = f'{f.filename} path is not like root/folder/pic.jpg'
            # return result

        root, folder, filename = f.filename.split('/')
        result['root'] = root

        if filename.split('.')[-1].lower() != 'jpg':
            result["success"] = False
            result['error'] = f'{filename} is not jpg file'
            return result

        if not os.path.exists(f'{app.config["UPLOAD_FOLDER"]}'):
            os.mkdir(f'{app.config["UPLOAD_FOLDER"]}')
        if not os.path.exists(f'{app.config["UPLOAD_FOLDER"]}/{root}'):
            os.mkdir(f'{app.config["UPLOAD_FOLDER"]}/{root}')
        if not os.path.exists(f'{app.config["UPLOAD_FOLDER"]}/{root}/{folder}'):
            os.mkdir(f'{app.config["UPLOAD_FOLDER"]}/{root}/{folder}')

        f.save(f'{app.config["UPLOAD_FOLDER"]}/{root}/{folder}/{filename}')

        if not result["data"].get(folder):
            result["data"][folder] = []
        result["data"][folder].append(filename)
    else:
        result["success"] = True
    return result


def process_folder(root, folder, files):
    answer = np.zeros(3, dtype=float)
    amount = 0
    for f in files:
        image_path = f'{app.config["UPLOAD_FOLDER"]}/{root}/{folder}/{f}'
        pred = classifier.predict(image_path)
        if pred["face_status"]:
            amount += 1
            answer += np.array(pred["data"])
    if amount == 0:
        return "face not detected"
    answer /= amount
    return {
        "fatness": "fat" if answer[0] > 25 else "skinny",
        "age": int(answer[1]),
        "sex": "female" if answer[2] < 0.05 else "male"
    }


@app.route('/upload', methods=['POST'])
def upload_handler():
    saved_files = save_files(request)
    response = {
        'status': 'error',
        'error': saved_files['error'],
        'data': {}
    }
    if not saved_files['success']:
        if os.path.exists(f'{app.config["UPLOAD_FOLDER"]}'):
            shutil.rmtree(f'{app.config["UPLOAD_FOLDER"]}', ignore_errors=True)
        return response
    response['status'] = 'success'
    response['error'] = None
    for folder in saved_files['data'].keys():
        tmp = process_folder(
            saved_files['root'],
            folder,
            saved_files['data'][folder]
        )
        response['data'][folder] = tmp
    if os.path.exists(f'{app.config["UPLOAD_FOLDER"]}'):
        shutil.rmtree(f'{app.config["UPLOAD_FOLDER"]}')
    return response


app.run(host='0.0.0.0', port='8002')
