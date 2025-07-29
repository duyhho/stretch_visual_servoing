import subprocess

# List the COMMANDS to run, not the filenames
scripts_to_run = [
    "stretch_free_robot_process.py",
    "stretch_robot_home.py"
]

print("--- Starting robot initialization sequence ---")

for command in scripts_to_run:
    print(f"\n>>> Executing command: {command}")
    try:
        # Run the command directly
        subprocess.run([command], check=True)
        print(f"<<< Finished: {command}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"!!! Error running {command}: {e}")
        break # Stop if a command fails

print("\n--- Robot initialization complete ---")