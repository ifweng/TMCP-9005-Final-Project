#!/bin/bash
# This script installs EternaFold, ViennaRNA, IPKnot and arnie packages in /tools directory
# Prerequisites:
# - git must be preinstalled
# - cmake version 3.23.1 must be preinstalled

set -e # Exit immediately if a command exits with a non-zero status
set -u # Treat unset variables as an error

echo "Creating tools directory..."
mkdir -p tools
cd tools 
tools_path=$(pwd)
echo "Installing tools in: $tools_path"

# Function to print section headers
print_header() {
  echo "=============================================="
  echo "  $1"
  echo "=============================================="
}

# Function to handle errors
handle_error() {
  echo "ERROR: Installation failed at $1"
  exit 1
}

# Install ViennaRNA package (version 2.6.4)
print_header "Installing ViennaRNA 2.6.4"
echo "Downloading ViennaRNA..."
wget https://www.tbi.univie.ac.at/RNA/download/sourcecode/2_6_x/ViennaRNA-2.6.4.tar.gz || handle_error "ViennaRNA download"

echo "Extracting ViennaRNA..."
tar -xvf ViennaRNA-2.6.4.tar.gz
rm ViennaRNA-2.6.4.tar.gz
mkdir ViennaRNA
cd ViennaRNA-2.6.4

# Follow the steps specified in official tutorial 
# https://github.com/ViennaRNA/ViennaRNA/blob/master/INSTALL
echo "Configuring ViennaRNA..."
./configure --prefix=$tools_path/ViennaRNA || handle_error "ViennaRNA configure"

echo "Building ViennaRNA (this may take a while)..."
make || handle_error "ViennaRNA make"
make check || handle_error "ViennaRNA make check"
make install || handle_error "ViennaRNA make install"
make installcheck || handle_error "ViennaRNA make installcheck"

cd $tools_path

# Install EternaFold (version 1.3.1)
print_header "Installing EternaFold"
echo "Cloning EternaFold repository..."
git clone https://github.com/eternagame/EternaFold.git || handle_error "EternaFold git clone"
cd EternaFold/src
echo "Building EternaFold..."
make || handle_error "EternaFold make"

cd $tools_path

# Install IPknot (installation info: https://github.com/satoken/ipknot)
print_header "Installing GLPK solver for IPknot"
# Set up GLPK solver (from https://www.gnu.org/software/glpk/) (version 5.0)
echo "Downloading GLPK 5.0..."
wget https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz || handle_error "GLPK download"
echo "Extracting GLPK..."
tar -xvf glpk-5.0.tar.gz
rm glpk-5.0.tar.gz

# Follow the install instructions in INSTALL file in glpk-5.0 directory
mkdir glpk
cd glpk-5.0
echo "Configuring GLPK..."
./configure --prefix=$tools_path/glpk || handle_error "GLPK configure"
echo "Building GLPK..."
make || handle_error "GLPK make"
make check || handle_error "GLPK make check"
make install || handle_error "GLPK make install"

cd $tools_path

# Install IPknot from https://github.com/satoken/ipknot (version 1.1.0)
print_header "Installing IPknot 1.1.0"
echo "Cloning IPknot repository..."
git clone https://github.com/satoken/ipknot.git || handle_error "IPknot git clone"
mv ipknot ipknot-1.1.0
cd ipknot-1.1.0
export PKG_CONFIG_PATH=$tools_path/ViennaRNA/lib/pkgconfig:$PKG_CONFIG_PATH
mkdir build && cd build
echo "Configuring IPknot with cmake..."
cmake -DCMAKE_BUILD_TYPE=Release .. || handle_error "IPknot cmake"

# As GLPK is installed in non-root directory, modify CMakeCache.txt to specify correct paths
echo "Modifying CMakeCache.txt with correct GLPK paths..."
sed -i "s#GLPK_INCLUDE_DIR:PATH=.*#GLPK_INCLUDE_DIR:PATH=$tools_path/glpk/include#g" CMakeCache.txt
sed -i "s#GLPK_LIBRARY:FILEPATH=.*#GLPK_LIBRARY:FILEPATH=$tools_path/glpk/lib/libglpk.a#g" CMakeCache.txt
sed -i "s#GLPK_ROOT_DIR:PATH=.*#GLPK_ROOT_DIR:PATH=$tools_path/glpk#g" CMakeCache.txt

echo "Building IPknot..."
cmake --build . || handle_error "IPknot build"
echo "Installing IPknot..."
cmake --install . --prefix $tools_path/ipknot || handle_error "IPknot install"

cd $tools_path
mkdir -p tmp

# Install arnie
print_header "Installing arnie"
echo "Cloning arnie repository..."
git clone https://github.com/DasLab/arnie.git || handle_error "arnie git clone"
cd arnie

# Make arnie config file
echo "Creating arnie configuration file..."
cat > arnie_config.txt << EOF
vienna_2: $tools_path/ViennaRNA-2.6.4/src/bin
contrafold_2: None
rnastructure: None
rnasoft: None
vfold: None
nupack: None
eternafold: $tools_path/EternaFold/src
ipknot: $tools_path/ipknot/bin
TMP: $tools_path/tmp
EOF

print_header "Installation Complete"
echo "The following tools have been installed:"
echo "- ViennaRNA 2.6.4: $tools_path/ViennaRNA"
echo "- EternaFold: $tools_path/EternaFold"
echo "- GLPK 5.0: $tools_path/glpk"
echo "- IPknot 1.1.0: $tools_path/ipknot"
echo "- arnie: $tools_path/arnie"
echo ""
echo "arnie configuration has been set up at: $tools_path/arnie/arnie_config.txt"