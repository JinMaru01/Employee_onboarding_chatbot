from flask import Flask
from routes import register_routes

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__, static_folder='static')
    
    # Register all routes
    register_routes(app)
    
    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)