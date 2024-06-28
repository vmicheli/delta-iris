fps=5
header=1
save_mode=0
mode="agent_in_env"

while [ "$1" != "" ]; do
    case $1 in
        -f | --fps )
            shift
            fps=$1
            ;;
        -h | --noheader )
            header=0
            ;;
        -s | --save-mode )
            save_mode=1
            ;;
        -a | --agent-world-model )
            mode="agent_in_world_model"
            ;;
        -e | --episode )
            mode="episode_replay"
            ;;
        -w | --world-model )
            mode="play_in_world_model"
            ;;
        * )
            break
    esac
    shift
done

python src/play.py hydra.run.dir=. hydra.output_subdir=null +mode="${mode}" +fps="${fps}" +header="${header}" +save_mode="${save_mode}" "$@"
