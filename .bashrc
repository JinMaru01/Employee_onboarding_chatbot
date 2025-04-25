alias activate='source ./env/Scripts/activate'
alias exec='python ./python/app.py'

run_flask(){
    source ./env/Scripts/activate
    python ./flask_app.py
}

run_app(){
    source ./env/Scripts/activate
    python ./app.py
}