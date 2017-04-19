from keras.utils import plot_model
from keras.models import load_model
from keras.models import Sequential

model = load_model("my_model.h5py")
plot_model(model, to_file='model.png')