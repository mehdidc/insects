
# Folder of the ramp
FOLDER="/home/mcherti/ramp_pollenating_insects/databoard_pollenating_insects_2170"
# Command to launch each time we connect to a server in reims
INIT_CMD="source /etc/profile;source ~/.myworkrc"
# Gate 
GATE=mcherti@romeo1.univ-reims.fr
# GPU machines
MACHINES=$(./generate_machine_numbers.sh training)

launch(){

    # 1) Launch the server on the gate
    PREFIX="ssh $GATE"
    PREFIX=$PREFIX' "'"screen -m -d -S server bash -c "
    CMD="$INIT_CMD;cd $FOLDER;python databoard/machine_parallelism.py server"
    WHOLE=$PREFIX"'"$CMD"'"'"'
    echo $WHOLE
    eval $WHOLE

    # 2) Launch the clients on GPU machine

    for machine in $MACHINES; do
        echo $machine
        CMD="$INIT_CMD;cd $FOLDER;python databoard/machine_parallelism.py client --host=romeo1"
        CMD="ssh $GATE 'screen -m -d	-S client_$machine ssh $machine ""\""$CMD"\"'"
        eval $CMD
    done

}

# 3) Run fab command


run(){

    CMD="fab train_test"
    #-- Uncomment this to train!
    ssh $GATE "$INIT_CMD;cd $FOLDER;$CMD"
}


clean(){

    # 4) Disconnect

    # 4-1) disconnect the GPU clients
    for machine in $MACHINES; do
    ssh -t $GATE ssh $machine pkill -f "machine_parallelism.py"
    ssh -t $GATE screen -S "client_$machine" -X quit
    done

    # 4-2) disconnect the server on the gate
    ssh -t $GATE ssh $machine pkill -f "machine_parallelism.py"
    ssh $GATE "screen -S server -X quit"

}


rsync(){
    # 5) rsync

    # --uncomment this to rsync
    SOURCE=$GATE:$FOLDER
    DEST=root@onevm-85.lal.in2p3.fr:?
    #rsync --update -raz --progress $SOURCE $DEST
}


if [ $# -eq 0 ]; then
    echo "No arguments provided, please choose between run and clean_only"
    exit 1
fi

case $1 in

    run)
        launch
        run
        clean
        #rsync    
        ;;
    clean_only)
        clean
        ;;
esac

