#! bin/bash

# /usr/local/lib/python3.7/dist-packages/keras_vggface/models.py in <module>()
#      18 from keras import backend as K
#      19 from keras_vggface import utils
# ---> 20 from keras.engine.topology import get_source_inputs <---- from keras.utils.layer_utils import get_source_inputs
#      21 import warnings
#      22 from keras.models import Model

# ModuleNotFoundError: No module named 'keras.engine.topology'


echo "Installing packages"
. env/bin/activate

python3.7 -m pip install tensorflow==2.8.2
python3.7 -m pip install numpy>=1.20.0 
python3.7 -m pip install pandas==0.25.0 
python3.7 -m pip install Keras>=2.8.1 
python3.7 -m pip install matplotlib==3.3.0 
python3.7 -m pip install mtcnn==0.1.0 
python3.7 -m pip install keras_vggface==0.6 
python3.7 -m pip install scikit_learn==0.23.1
python3.7 -m pip install keras_applications
python3.7 -m pip install flask
