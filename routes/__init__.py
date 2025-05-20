from .home import home_bp
from .api import api_bp

def register_routes(app):
    app.register_blueprint(home_bp)
    app.register_blueprint(api_bp)