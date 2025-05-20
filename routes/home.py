from flask import Blueprint, render_template

home_bp = Blueprint('home', __name__)

@home_bp.route("/", methods=['GET'])
def home():
    """
    Render the home page.
    
    Returns:
        HTML template for the home page
    """
    return render_template("index.html")