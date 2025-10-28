#!/bin/bash

# Create directory structure
mkdir -p src/ec src/mo src/spc src/plotting src/utils
touch src/__init__.py src/ec/__init__.py src/mo/__init__.py src/spc/__init__.py src/plotting/__init__.py src/utils/__init__.py

# Function to safely move files
safe_move() {
    if [ -f "$1" ]; then
        mv "$1" "$2"
        echo "Moved: $1 -> $2"
    else
        echo "Not found: $1"
    fi
}

# Move EC files - try different possible locations
safe_move "EC/constants.py" "src/ec/constants.py"
safe_move "constants.py" "src/ec/constants.py"

safe_move "EC/Func_read_data.py" "src/ec/func_read_data.py"
safe_move "Func_read_data.py" "src/ec/func_read_data.py"

safe_move "EC/Func_despike_data.py" "src/ec/func_despike_data.py"
safe_move "Func_despike_data.py" "src/ec/func_despike_data.py"

safe_move "EC/sensor_info.py" "src/ec/sensor_info.py"
safe_move "sensor_info.py" "src/ec/sensor_info.py"

safe_move "EC/Func_DR.py" "src/ec/func_dr.py"
safe_move "Func_DR.py" "src/ec/func_dr.py"

safe_move "EC/Func_MRFD.py" "src/ec/func_mrfd.py"
safe_move "Func_MRFD.py" "src/ec/func_mrfd.py"

# Move MO files
safe_move "MO/Func_MO.py" "src/mo/func_mo.py"
safe_move "Func_MO.py" "src/mo/func_mo.py"

# Move SPC files
safe_move "SPC/normalize.py" "src/spc/normalize.py"
safe_move "normalize.py" "src/spc/normalize.py"

# Move plotting files
safe_move "plotting/Funcs_plots.py" "src/plotting/funcs_plots.py"
safe_move "Funcs_plots.py" "src/plotting/funcs_plots.py"

echo "Reorganization complete!"
