# cli_interface.py
import os
import sys
from workflow_engine import WorkflowEngine
from dotenv import load_dotenv
import datetime
import warnings

suppress_warnings = True
if suppress_warnings:
    warnings.filterwarnings("ignore")  

def print_header():
    version = "v1.0.0"
    developer = "d3bug1t"
    institution = "IIT Kanpur"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"""
 ███╗   ███╗ ██████╗ ███████╗    ███████╗██╗███╗   ███╗
 ████╗ ████║██╔═══██╗██╔════╝    ██╔════╝██║████╗ ████║
 ██╔████╔██║██║   ██║█████╗      ███████╗██║██╔████╔██║
 ██║╚██╔╝██║██║   ██║██╔══╝      ╚════██║██║██║╚██╔╝██║
 ██║ ╚═╝ ██║╚██████╔╝██║         ███████║██║██║ ╚═╝ ██║
 ╚═╝     ╚═╝ ╚═════╝ ╚═╝         ╚══════╝╚═╝╚═╝     ╚═╝
 
        MOF Discovery & RASPA Simulation Assistant
        -------------------------------------------------
        Version      : {version}
        Developed by : {developer}
        Institution  : {institution}
        Session      : {timestamp}
        RASPA_DIR    : {os.environ.get("RASPA_DIR", "Not Set")}
""")


def main():

    print_header()

    
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if "RASPA_DIR" in os.environ:
        os.environ["PATH"] = os.path.join(os.environ["RASPA_DIR"], "bin") + os.pathsep + os.environ.get("PATH", "")

    api_key = GOOGLE_API_KEY # Set if needed
    engine = WorkflowEngine(api_key=api_key)

    while True:

        print("\nEnter your query (or type 'exit' to quit):")
        user_input = input("> ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Exiting. Goodbye!")
            sys.exit(0)

        print("\n==============================")
        print("🔎 Starting Workflow")
        print("==============================\n")

        try:
            engine.run(user_input, mock=False)
        except Exception as e:
            print("\n❌ Workflow failed:")
            print(str(e))

        print("\n==============================")
        print("✅ Workflow Finished")
        print("==============================\n")


if __name__ == "__main__":
    main()
