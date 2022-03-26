import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.backend import expand_dims, repeat_elements, log, exp
import numpy as np
import h5py
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import os


gpus = tf.config.experimental.list_physical_devices('GPU')

if len(gpus) == 0:
    print("GPU required!")
    exit(1)


class DataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, paths, batch_size=20, dim=(64, 64),
                 shuffle=True, input_steps=16, output_steps=8, skip_steps=1):
        """Initialization"""
        self.dim = dim
        self.file_batch_size = 2048
        self.batch_size = batch_size
        self.n_batches = self.file_batch_size // self.batch_size
        self.file_paths = paths
        self.shuffle = shuffle
        self.current_file_loaded = (None, None)
        self.steps = (input_steps, output_steps, skip_steps)
        self.indexes = np.arange(len(self.file_paths) * int(self.file_batch_size / self.batch_size))
        self.on_epoch_end()

    def load_file(self, i):
        with h5py.File(self.file_paths[i], "r") as f:
            self.current_file_loaded = i, (f["data/X"][:], f["data/y"][:])

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.file_paths) * self.n_batches

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        file_index = index // self.n_batches
        current_file_index, _ = self.current_file_loaded
        if file_index != current_file_index:
            self.load_file(file_index)
        _, (X, y) = self.current_file_loaded
        index_inside_file = index % self.n_batches
        i = index_inside_file * self.batch_size
        inp, out, skip = self.steps
        X = X[i:(i + self.batch_size), -inp:].astype(np.float32)
        y = y[:, skip-1::skip]
        y = y[i:(i + self.batch_size), :out].astype(np.float32)
        y[y > 1.0] = np.float32(1.0)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.file_paths)
            self.current_file_loaded = (None, None)



def get_compiled_model(input_steps, output_steps, lr, filters):

    inp = layers.Input(shape=(input_steps, 64, 64, 1))

    # Probar normalizar dividiendo entre 300
    # Probar normalizar log(x+1)
    x = Lambda(lambda x: log(x + 1))(inp)
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(3, 3),
#        return_sequences=True,
        padding="same",
        activation="relu",
    )(x)
    x = Lambda(lambda x: repeat_elements(expand_dims(x, axis=1), output_steps, 1))(x)
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.ConvLSTM2D(
         filters=filters,
         kernel_size=(1, 1),
         #padding="same",
         return_sequences=True,
         activation="relu",
     )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)
    # x = Lambda(exp)(x)
    model = Model(inp, x)
    model.compile(
        loss=BinaryCrossentropy(), 
        optimizer=Adam(learning_rate=lr),
    )
    return model




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

def get_datasets(folder, splits, input_steps, output_steps, batch_size):
    files = np.array([os.path.join(folder, x) for x in os.listdir(folder)][:-1])
    train_files, val_files, test_files = split(files, splits)
    train_generator = DataGenerator(train_files, input_steps=input_steps, output_steps=output_steps, batch_size=batch_size)
    val_generator = DataGenerator(val_files, input_steps=input_steps, output_steps=output_steps, batch_size=batch_size)
    test_generator = DataGenerator(test_files, input_steps=input_steps, output_steps=output_steps, batch_size=batch_size)
    return (train_generator, val_generator, test_generator)

INPUT_STEPS = 16
OUTPUT_STEPS = 8
STEP_JUMP = 2


folder = "./datastores/glm-boxes-2048/"






model = get_compiled_model(INPUT_STEPS, OUTPUT_STEPS)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TerminateOnNaN





def infer_dataset_folder():
    try:
        p = os.path.join(os.getcwd(), "datastores")
        dirs = os.listdir(p)
        if len(dirs) != 1:
            raise ValueError("Can't infer dataset folder")
        return os.path.join(p, dirs[0])
    except Exception as e:
        print(e.with_traceback())
        return os.getcwd()

if __name__ == "__main__":
    from argparse import ArgumentParser
    train_val_test_split = (.7, .2, .1)
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--in_steps', type=int, default=16)
    parser.add_argument('--out_steps', type=int, default=8)
    parser.add_argument('--skip_steps', type=int, default=1)
    parser.add_argument('--filters', type=int, default=1)
    parser.add_argument('--dataset', type=str, default=infer_dataset_folder())

    args = parser.parse_args()
    (train_generator, val_generator, test_generator) = get_datasets(
        args.dataset, 
        train_val_test_split,
        args.in_steps,
        args.out_steps,
        args.batch_size,
    )

    model = get_compiled_model(
        args.in_steps,
        args.out_steps,
        args.lr,
        args.filters,
    )
    print(model.summary())
    filepath = "saved-model-{epoch:02d}-{val_loss:.6f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto', save_weights_only=False)
    early_stopping = EarlyStopping(monitor="val_loss", patience=3)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=2)
    terminate_nan = TerminateOnNaN()
    model.fit(
        train_generator,
        epochs=args.max_epochs,
        validation_data=val_generator,
        callbacks=[
            early_stopping,
            reduce_lr,
            checkpoint,
            terminate_nan,
        ],
    )
    model.evaluate(test_generator)


