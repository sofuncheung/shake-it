for i in `seq 0 500 5000`
do
    echo "Attack size: $i"
    mkdir attack_size_$i
    cp config.py attack_size_$i/
    cd attack_size_$i/
    echo "    attack_set_size = $i" >> config.py
    dir=$(pwd)
    echo "$dir"

    for s in `seq 1 1 5`
    do
        echo "submitting sample $s"
        addqueue -q gpushort -n 1x4 -s  /mnt/zfsusers/sofuncheung/.venv/pyhessian/bin/python ~/shake-it/main.py -p $dir -s $s
    done
    cd ..
done
