from flask import Flask, send_from_directory, request, jsonify, Response
from flask_cors import CORS, cross_origin
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, FeatureExtractionPipeline
import transformers
import torch
import os
import json
import numpy as np
import umap
import time
#model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device="gpu", quantize=True)
model = SentenceTransformer('all-MiniLM-L12-v2')
app = Flask(__name__, static_folder='./static')
CORS(app)

# RUGPULL
# Retrieval
# yoU
# Generate
# Plus
# yoU
# Leverage
# Learning


@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_file(path):
    return send_from_directory(app.static_folder, path)

@app.route('/post', methods=['POST'])
@cross_origin()
def post():
    data = request.json  # Parse JSON data from the request
    encoded = model.encode(data, show_progress_bar=True, device='mps', batch_size=128)
    bytes_data = encoded.tobytes()
    return Response(bytes_data, mimetype='application/octet-stream')

@app.route('/list-files', methods=['POST'])
@cross_origin()
def list_post_files():
    path = request.json['path']
    files = os.listdir(path)
    # Get rid of .DS_Store
    files = [f for f in files if not f.startswith('.')]
    return jsonify(files)


@app.route('/post-umap', methods=['POST'])
@cross_origin()
def post_umap():
    binary_data = np.array(request.data)
    # Convert to NumPy array assuming each element is a 32-bit float
    np_array = np.frombuffer(binary_data, dtype=np.float32)

    # Reshape the array
    # Adjust the shape calculation as per your requirements
    row_count = len(np_array) // 384
    np_matrix = np_array.reshape((row_count, 384))
    start_time = time.time()
    umap_data = umap.UMAP(n_neighbors=min(15, row_count), min_dist=0.0, n_components=2, n_jobs=4, low_memory=False, metric='cosine', verbose=True).fit_transform(np_matrix)
    print("UMAP took", time.time() - start_time, "seconds")
    return Response(umap_data.tobytes(), mimetype='application/octet-stream')

@app.route('/save', methods=['POST'])
@cross_origin()
def save():
    data = request.json['data']
    filename = request.json['filename']
    os.makedirs(f"cache/{filename}", exist_ok=True)
    #json.dump(data, open("data.json", "w"))
    with open(f"cache/{filename}/index.json", "w") as f:
        data_wout_embeddings = data.copy()
       # del data_wout_embeddings['embedding_u8']
       #del data_wout_embeddings['contextMinMax']
        #del data_wout_embeddings['results']
        json.dump(data_wout_embeddings, f)
    return Response("Saved", mimetype='text/plain')
    # Create directory for embeddings, cache/{filename}/embeddings.npy
        
        
    """  with open(f"cache/{filename}/embedding_u8.npy", "wb") as f:
        e = request.json['data']['embedding_u8']
        e = np.array(e, dtype=np.uint8)
        np.save(f, e)

    with open(f"cache/{filename}/embedding_ctx.npy", "wb") as f:
        e = request.json['data']['contextMinMax']
        e = np.array(e, dtype=np.float32)
        np.save(f, e)

    with open(f"cache/{filename}/results.npy", "wb") as f:
        e = request.json['data']['results']
        e = np.array(e, dtype=np.float32)
        np.save(f, e)

    return Response("Saved", mimetype='text/plain')"""

@app.route('/save-embeddings-u8', methods=['POST'])
@cross_origin()
def save_embeddings_u8():
    if 'file' in request.files:
        file = request.files['file']
        binary_data = file.read()
        # Convert to NumPy array assuming each element is a 32-bit float
        np_array = np.frombuffer(binary_data, dtype=np.uint8)

        # Reshape the array
        # Adjust the shape calculation as per your requirements
        row_count = len(np_array) // 384
        np_matrix = np_array.reshape((row_count, 384))
        file_path = request.form.get('path', '')
        with open(f"cache/{file_path}/embedding_u8.npy", "wb") as f:
            np.save(f, np_matrix)
        return Response("Saved", mimetype='text/plain')
    else:
        return Response("No file", mimetype='text/plain')
   



@app.route('/save-embeddings-ctx', methods=['POST'])
@cross_origin()
def save_embeddings_ctx():
    if 'file' in request.files:
        file = request.files['file']
        binary_data = file.read()
        # Convert to NumPy array assuming each element is a 32-bit float
        np_array = np.frombuffer(binary_data, dtype=np.float32)

        # Reshape the array
        # Adjust the shape calculation as per your requirements
        row_count = len(np_array) // 2
        np_matrix = np_array.reshape((row_count, 2))
        file_path = request.form.get('path', '')
        with open(f"cache/{file_path}/embedding_ctx.npy", "wb") as f:
            np.save(f, np_matrix)
        return Response("Saved", mimetype='text/plain')
    else:
        return Response("No file", mimetype='text/plain')
    
@app.route('/save-results', methods=['POST'])
@cross_origin()
def save_results():
    if 'file' in request.files:
        file = request.files['file']
        binary_data = file.read()
        # Convert to NumPy array assuming each element is a 32-bit float
        np_array = np.frombuffer(binary_data, dtype=np.float32)

        # Reshape the array
        # Adjust the shape calculation as per your requirements
        row_count = len(np_array) // 3
        np_matrix = np_array.reshape((row_count, 3))
        file_path = request.form.get('path', '')
        with open(f"cache/{file_path}/results.npy", "wb") as f:
            np.save(f, np_matrix)
        return Response("Saved", mimetype='text/plain')
    else:
        return Response("No file", mimetype='text/plain')


"""@app.route('/load', methods=['POST'])
@cross_origin()
def load():
    filename = request.json['filename']
    data = {}
    with open(f"cache/{filename}/embeddings.npy", "rb") as f:
        e = np.load(f)
        e = e.astype(np.float32)
        data['embeddings'] = e.tolist()

    with open("cache/" + filename + ".json", "r") as f:
        data_wout_embeddings = json.load(f)
        data.update(data_wout_embeddings)"""


@app.route('/load', methods=['POST'])
@cross_origin()
def load():
    filename = request.json['filename']
    data_wout_embeddings = {}
    with open(f"cache/{filename}/index.json", "r") as f:
        data_wout_embeddings.update(json.load(f))
    return jsonify(data_wout_embeddings)


"""@app.route('/load-embeddings', methods=['POST'])
@cross_origin()
def load_embeddings():
    filename = request.json['filename']
    with open(f"cache/{filename}/embeddings.npy", "rb") as f:
        e = np.load(f)
        e = e.astype(np.float32)

    return Response(e.tobytes(), mimetype='application/octet-stream')"""

@app.route('/load-embeddings-u8', methods=['POST'])
@cross_origin()
def load_embeddings_u8():
    filename = request.json['filename']
    with open(f"cache/{filename}/embedding_u8.npy", "rb") as f:
        e = np.load(f)
        e = e.astype(np.uint8)

    return Response(e.tobytes(), mimetype='application/octet-stream')
@app.route('/load-embeddings-ctx', methods=['POST'])
@cross_origin()
def load_embeddings_ctx():
    filename = request.json['filename']
    with open(f"cache/{filename}/embedding_ctx.npy", "rb") as f:
        e = np.load(f)
        e = e.astype(np.float32)

    return Response(e.tobytes(), mimetype='application/octet-stream')

@app.route('/load-results', methods=['POST'])
@cross_origin()
def load_results():
    filename = request.json['filename']
    with open(f"cache/{filename}/results.npy", "rb") as f:
        e = np.load(f)
        e = e.astype(np.float32)

    return Response(e.tobytes(), mimetype='application/octet-stream')
if __name__ == '__main__':
    app.run(host="localhost", port="3000", debug=True)