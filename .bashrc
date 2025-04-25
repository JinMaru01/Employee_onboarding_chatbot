run(){
    source ./env/Scripts/activate
    python "$@"
    deactivate
    clear
}