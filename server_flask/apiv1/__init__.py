from flask_restplus import Api
from flask import Blueprint

from .imageController import api as ns1

blueprint = Blueprint('api', __name__)
api = Api(blueprint, default_mediatype='application/json')

api.add_namespace(ns1)