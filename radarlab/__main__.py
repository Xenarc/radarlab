import sys


if __name__ == "__main__":
    match sys.argv[1:2]:
        case ["visualise"]:
            sys.argv.pop(1)
            from radarlab.visualise import main as visualiser_main
            visualiser_main()
        case ["diagnostics"]:
            sys.argv.pop(1)
            from radarlab.diagnostics import main as diagnostic_main
            diagnostic_main()
        case ["lab"]:
            sys.argv.pop(1)
            from radarlab.lab import lab as lab_main
            lab_main()
        case ["analysis"]:
            match sys.argv[2:3]:
                case ["target_altitude"]:
                    from radarlab.analysis.target_altitude import run as target_altitude_run
                    target_altitude_run()                   
                case ["sea_state"]:
                    from radarlab.analysis.sea_state import run as sea_state_run
                    sea_state_run()                   
                case ["frequency"]:
                    from radarlab.analysis.frequency import run as frequency_run
                    frequency_run()                   
                case ["displacement"]:
                    from radarlab.analysis.displacement import run as displacement_run
                    displacement_run()            
                case _:
                    sys.exit(f"usage: {sys.argv[0]} analysis {{target_altitude, sea_state, frequency, displacement}} [args...]")       

        case _:
            sys.exit(f"usage: {sys.argv[0]} {{visualise, diagnostics, lab, analysis}} [args...]")