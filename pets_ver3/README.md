# PETS Training

Run the training script with configurable seed and episode count.

## Setup (Windows)

```powershell
# Optional: create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Run

```powershell
# Run with seed 0 and 200 episodes
python pets_train.py --seed 0 --steps 200
```

If you prefer the workspace interpreter detected by VS Code, you can run:

```powershell
& "C:/Users/HP/AppData/Local/Programs/Python/Python310/python.exe" "C:/Users/HP/pets_ver3/pets_train.py" --seed 0 --steps 200
```

The script will print progress every 5 episodes, and summary stats at the end.
