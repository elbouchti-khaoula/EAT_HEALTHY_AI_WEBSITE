from app import create_app
import urllib
app = create_app()
app.config["SECRET_KEY"] = '79537d00f4834892986f09a100aa1edf'

if __name__ == '__main__':
    from waitress import serve
    serve(app)
