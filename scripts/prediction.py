import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np
import h5py


class DataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, paths, b_size=64, dim=(64, 64), n_channels=1,
                 n_classes=10, shuffle=True, k=8):
        """Initialization"""
        self.dim = dim
        self.file_batch_size = 1024
        self.batch_size = b_size
        self.factor = self.file_batch_size // b_size
        self.file_paths = paths
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.current_file_loaded = (None, None)
        self.k = k
        self.indexes = np.arange(len(self.file_paths) * int(self.file_batch_size / self.batch_size))
        self.on_epoch_end()

    def load_file(self, i):
        f = h5py.File(self.file_paths[i], "r")
        self.current_file_loaded = i, f["flash_extent_density/train"][:]

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.file_paths) * self.factor // 2
        # TODO: Arreglar esto

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        file_index = index // self.factor
        current_file_index, _ = self.current_file_loaded
        if file_index != current_file_index:
            self.load_file(file_index)
        _, current_data = self.current_file_loaded
        index_inside_file = index % self.factor
        i = index_inside_file * self.batch_size
        data = current_data[i:(i + self.batch_size)]
        data /= data.max()
        X = data[:, :self.k]
        y = data[:, self.k:]
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.file_paths)
            self.current_file_loaded = (None, None)

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam

def get_compiled_model():
    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=(None, 64, 64, 1))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)
    # Next, we will build the complete model and compile it.

    model = Model(inp, x)
    model.compile(
        loss=binary_crossentropy, optimizer=Adam(),
    )
    return model




import os
folder = "./data/exp_pro/GLM-L2-LCFA_8km_5m_boxes/2019/h5"
files = np.array([os.path.join(folder, x) for x in os.listdir(folder)][:-1])

train_val_test_split = (.7, .2, .1)

def split(arr, splits):
    arr = np.array(arr)
    idxs = np.arange(len(arr))
    np.random.shuffle(idxs)
    n_splits = list(int(len(arr)*x) for x in splits)
    datasets = []
    start = 0
    n_splits[-1] += len(arr) - sum(n_splits)
    n_splits = tuple(n_splits)
    for split in n_splits:
        datasets.append(arr[idxs[start:start+split]])
        start += split
    return tuple(datasets)

train_files, val_files, test_files = split(files, train_val_test_split)

train_generator = DataGenerator(train_files)
val_generator = DataGenerator(val_files)
test_generator = DataGenerator(test_files)

strategy = tensorflow.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model = get_compiled_model()


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# Define some callbacks to improve training.

filepath = "saved-model-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto', save_weights_only=False)
early_stopping = EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5)
# Define modifiable training hyperparameters.
epochs = 20
# Fit the model to the training data.
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr, checkpoint],
)

# Test the model on all available devices.
model.evaluate(test_generator)
