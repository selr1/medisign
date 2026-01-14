### Setup

```bash
# install miniconda from the AUR
 yay -S miniconda3 
# initialize (bash)
 source /opt/miniconda3/bin/activate
 conda init bash
# initialize (fish)
 source /opt/miniconda3/etc/fish/conf.d/conda.fish
 conda init fish
# create & activate .venv
 conda create -n MediSign python=3.10 -y
 conda activate MediSign
# install requirements
 pip install mediapipe opencv-python pandas numpy scikit-learn joblib 

```

