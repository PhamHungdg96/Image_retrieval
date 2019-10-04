from flask import Flask
from flask_cors import CORS
from apiv1 import blueprint as api_v1

app = Flask(__name__, static_url_path='')
CORS(app,resources={r"/api/v1/*": {"origins": "*"}})
app.register_blueprint(api_v1,url_prefix='/api/v1')
app.run(debug=True)