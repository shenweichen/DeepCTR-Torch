import copy
import os

import numpy as np
import torch


class Callback(object):
    def __init__(self):
        """Initialize callback state."""
        self.model = None
        self.params = {}

    def set_model(self, model):
        self.model = model

    def set_params(self, params):
        self.params = params or {}

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass


class CallbackList(object):
    def __init__(self, callbacks=None):
        """Create a callback container."""
        self.callbacks = list(callbacks or [])
        self.model = None
        self.params = {}

    def append(self, callback):
        self.callbacks.append(callback)

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def set_params(self, params):
        self.params = params or {}
        for callback in self.callbacks:
            callback.set_params(self.params)

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs=logs)


class History(Callback):
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}
        if self.model is not None:
            self.model.history = self

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    ):
        """Create an early-stopping callback."""
        super(EarlyStopping, self).__init__()
        self.monitor = monitor
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights

        if mode not in {"auto", "min", "max"}:
            raise ValueError("mode should be one of {'auto', 'min', 'max'}")

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if "acc" in self.monitor or self.monitor.endswith("auc") or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.inf if self.monitor_op == np.less else -np.inf

    def _is_improvement(self, current, best):
        if self.monitor_op == np.less:
            return current < (best - self.min_delta)
        return current > (best + self.min_delta)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self._is_improvement(current, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights and self.model is not None:
                self.best_weights = copy.deepcopy(self.model.state_dict())
            return

        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch + 1
            if self.model is not None:
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.load_state_dict(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print("Epoch %05d: early stopping" % self.stopped_epoch)


class ModelCheckpoint(Callback):
    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        period=1,
    ):
        """Create a model-checkpoint callback."""
        super(ModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in {"auto", "min", "max"}:
            raise ValueError("mode should be one of {'auto', 'min', 'max'}")

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            if "acc" in self.monitor or self.monitor.endswith("auc") or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                self.best = -np.inf
            else:
                self.monitor_op = np.less
                self.best = np.inf

    def _save(self, filepath):
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if self.save_weights_only:
            torch.save(self.model.state_dict(), filepath)
        else:
            torch.save(self.model, filepath)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return

        self.epochs_since_last_save = 0
        filepath = self.filepath.format(epoch=epoch + 1, **logs)

        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                if self.verbose > 0:
                    print("Can save best model only with %s available, skipping." % self.monitor)
                return
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(
                        "Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
                        % (epoch + 1, self.monitor, self.best, current, filepath)
                    )
                self.best = current
                self._save(filepath)
            elif self.verbose > 0:
                print("Epoch %05d: %s did not improve from %0.5f" % (epoch + 1, self.monitor, self.best))
            return

        if self.verbose > 0:
            print("Epoch %05d: saving model to %s" % (epoch + 1, filepath))
        self._save(filepath)
