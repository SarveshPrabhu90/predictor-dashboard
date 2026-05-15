"""Flask application factory."""

from flask import Flask


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = testing

    from . import routes
    app.register_blueprint(routes.bp)

    return app
