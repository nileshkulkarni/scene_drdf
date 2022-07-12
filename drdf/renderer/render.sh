blenderExec=$1
blendFile=$2
renderFile=$3
objpath=$4
pngpath=$5
az=$6
el=$7
dist=$8
upsamp=$9
theta=${10}
$blenderExec $blendFile  -noaudio --background --python $renderFile -- $objpath $pngpath $az $el $dist $upsamp $theta  > /dev/null
