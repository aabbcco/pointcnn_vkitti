#!/usr/bin/env bash
#!/usr/bin/env bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/aabbcco/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/aabbcco/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/aabbcco/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/aabbcco/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

gpu=
setting=
models_folder="../../models/seg/"
train_files="../../data/vkitti3d/train_data_files.txt"
val_files="../../data/vkitti3d/val_data_files.txt"

usage() { echo "train/val pointcnn_seg with -g gpu_id -x setting options"; }

gpu_flag=0
setting_flag=0
while getopts g:x:h opt; do
  case $opt in
  g)
    gpu_flag=1;
    gpu=$(($OPTARG))
    ;;
  x)
    setting_flag=1;
    setting=${OPTARG}
    ;;
  h)
    usage; exit;;
  esac
done

shift $((OPTIND-1))

if [ $gpu_flag -eq 0 ]
then
  echo "-g option is not presented!"
  usage; exit;
fi

if [ $setting_flag -eq 0 ]
then
  echo "-x option is not presented!"
  usage; exit;
fi

if [ ! -d "$models_folder" ]
then
  mkdir -p "$models_folder"
fi

echo "Train/Val with setting $setting on GPU $gpu!"




CUDA_VISIBLE_DEVICES=$gpu python3 ../train_val_seg.py -t $train_files -v $val_files -s $models_folder  -m pointcnn_seg -x $setting > $models_folder/pointcnn_seg_$setting.txt 2>&1 &