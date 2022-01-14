#!/bin/bash
function cpu() {
    NUM=1
    while [ $NUM -le 3 ]; do
        util=`vmstat |awk '{if(NR==3)print 100-$15"%"}'`
        user=`vmstat |awk '{if(NR==3)print $13"%"}'`
        sys=`vmstat |awk '{if(NR==3)print $14"%"}'`
        iowait=`vmstat |awk '{if(NR==3)print $16"%"}'`
        echo "CPU - EFF: $util , Wait IO EFF: $iowait"
        let NUM++
        sleep 1
    done
}

function memory() {
    total=`free -m |awk '{if(NR==2)printf "%.1f",$2/1024}'`
    used=`free -m |awk '{if(NR==2) printf "%.1f",($2-$NF)/1024}'`
    available=`free -m |awk '{if(NR==2) printf "%.1f",$NF/1024}'`
    echo "MEM - Total_Size: ${total}G , USED: ${used}G , FREE: ${available}G"
}

function disk() {
    fs=$(df -h |awk '/^\/dev/{print $1}')
    for p in $fs; do
        mounted=$(df -h |awk '$1=="'$p'"{print $NF}')
        size=$(df -h |awk '$1=="'$p'"{print $2}')
        used=$(df -h |awk '$1=="'$p'"{print $3}')
        used_percent=$(df -h |awk '$1=="'$p'"{print $5}')
        echo "Hardware - mount: $mounted , Total_Size: $size , USED: $used , FREE: $used_percent"
    done
}

function tcp_status() {
    summary=$(ss -antp |awk '{status[$1]++}END{for(i in status) printf i":"status[i]" "}')
    echo "TCP Status - $summary"
}

cpu
memory
disk
tcp_status