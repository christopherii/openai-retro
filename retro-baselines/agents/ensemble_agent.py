from rainbow_agent import main as rainbow_main
from jerk_agent import main as jerk_main

if __name__ == '__main__':
    try:
        rainbow_main()
        jerk_main()
    except gre.GymRemoteError as exc:
        print('exception', exc)