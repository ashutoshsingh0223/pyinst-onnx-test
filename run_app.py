import os
from subprocess import Popen, PIPE, STDOUT
import sys
import time
# import webbrowser

# import streamlit

# import streamlit.web.cli as stcli

# def resolve_path(path):
#     resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
#     return resolved_path


# def main():
#     path_to_main = os.path.join(os.path.dirname(__file__), "main.py")

#     # Running streamlit server in a subprocess and writing to log file
#     proc = Popen(
#         [
#             "streamlit",
#             "run",
#             path_to_main,
#             # The following option appears to be necessary to correctly start the streamlit server,
#             # but it should start without it. More investigations should be carried out.
#             "--server.headless=true",
#             "--global.developmentMode=false",
#             "--server.enableXsrfProtection=false"
#         ],
#         stdin=PIPE,
#         stdout=PIPE,
#         stderr=STDOUT,
#         text=True,
#     )
#     proc.stdin.close()

#     # Force the opening (does not open automatically) of the browser tab after a brief delay to let
#     # the streamlit server start.
#     time.sleep(3)
#     webbrowser.open("http://localhost:8501")

#     while True:
#         s = proc.stdout.read()
#         print(s)
#         if not s:
#             break
#         print(s, end="")

#     proc.wait()


# if __name__ == "__main__":
    # main()

if __name__ == "__main__":
    path_to_main = os.path.join(os.path.dirname(__file__), "main.py")
    print(os.listdir(os.path.dirname(__file__)))
    # sys.argv = [
    #     "./env/bin/python3.10",
    #     "-m"
    #     "streamlit",
    #     "run",
    #     path_to_main,
    #     "--global.developmentMode=false",
    # ]
    # sys.exit(stcli.main())

    proc = Popen(
        [
            "./env/bin/python3.10",
            "-m",
            "streamlit",
            "run",
            path_to_main,
            # The following option appears to be necessary to correctly start the streamlit server,
            # but it should start without it. More investigations should be carried out.
            "--server.headless=true",
            "--global.developmentMode=false",
            "--server.enableXsrfProtection=false"
        ],
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
    )
    proc.stdin.close()

    while True:
        s = proc.stdout.read()
        if not s:
            break
        print(s, end="")

    proc.wait()