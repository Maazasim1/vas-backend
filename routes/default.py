from flask import Blueprint

bp = Blueprint('default', __name__)

@bp.route('/', methods=['GET'])
def default_route():
    return "this is the vas backend updated by V1.3"
