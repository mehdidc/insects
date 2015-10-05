#!/bin/sh
PREFIX=romeo

if [ $# -eq 0 ]; then
    echo "No arguments provided, please choose between training and test or ramp"
    exit 1
fi


training(){
    # training
    echo "$PREFIX"13
    echo "$PREFIX"16
    for v in $(seq 123 140); do
        echo "$PREFIX$v"
    done
}


test(){
    # test
    echo "$PREFIX"18
    echo "$PREFIX"37
}
#ramp
ramp(){
    for v in $(seq 16 18); do
        echo "$PREFIX$v"
    done
    for v in $(seq 27 42); do
        echo "$PREFIX$v"
    done
    for v in $(seq 58 120); do
        echo "$PREFIX$v"
    done
    for v in $(seq 123 140); do
        echo "$PREFIX$v"
    done
}

case $1 in

    training)
        training
        ;;
    test)
        test
        ;;
    ramp)
        ramp
esac

