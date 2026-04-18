# -*- coding: utf-8 -*-
import torch
import pytest

from deepctr_torch.callbacks import Callback, CallbackList, EarlyStopping, History, ModelCheckpoint


class ProbeCallback(Callback):
    def __init__(self):
        """Initialize probe callback state."""
        super(ProbeCallback, self).__init__()
        self.events = []

    def on_train_begin(self, logs=None):
        self.events.append(("train_begin", logs))

    def on_train_end(self, logs=None):
        self.events.append(("train_end", logs))

    def on_epoch_begin(self, epoch, logs=None):
        self.events.append(("epoch_begin", epoch, logs))

    def on_epoch_end(self, epoch, logs=None):
        self.events.append(("epoch_end", epoch, logs))


class TinyModel(torch.nn.Module):
    def __init__(self):
        """Initialize a tiny torch module for callback tests."""
        super(TinyModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.stop_training = False


def test_callback_and_callback_list_flow():
    cb = Callback()
    cb.set_model(TinyModel())
    cb.set_params(None)
    cb.on_train_begin()
    cb.on_epoch_begin(0)
    cb.on_epoch_end(0)
    cb.on_train_end()

    probe_1 = ProbeCallback()
    probe_2 = ProbeCallback()
    cb_list = CallbackList([probe_1])
    cb_list.append(probe_2)
    model = TinyModel()
    cb_list.set_model(model)
    cb_list.set_params(None)
    cb_list.on_train_begin(logs={"phase": "train"})
    cb_list.on_epoch_begin(1, logs={"loss": 0.2})
    cb_list.on_epoch_end(1, logs={"loss": 0.1})
    cb_list.on_train_end(logs={"done": True})

    assert probe_1.model is model and probe_2.model is model
    assert probe_1.params == {} and probe_2.params == {}
    assert ("train_begin", {"phase": "train"}) in probe_1.events
    assert ("train_end", {"done": True}) in probe_2.events


def test_history_records_logs():
    history = History()
    model = TinyModel()
    history.set_model(model)
    history.on_train_begin()
    history.on_epoch_end(0, {"loss": 0.3, "acc": 0.8})
    history.on_epoch_end(1, {"loss": 0.2, "acc": 0.9})

    assert model.history is history
    assert history.epoch == [0, 1]
    assert history.history["loss"] == [0.3, 0.2]
    assert history.history["acc"] == [0.8, 0.9]


def test_early_stopping_paths(capsys):
    with pytest.raises(ValueError):
        EarlyStopping(mode="unsupported")

    # Cover baseline/min branch + _is_improvement(min)
    es_min = EarlyStopping(monitor="val_loss", mode="min", baseline=0.5)
    es_min.on_train_begin()
    assert es_min.best == 0.5
    assert es_min._is_improvement(0.4, es_min.best)

    # Cover auto/max branch + restore-best-weights path
    model = TinyModel()
    with torch.no_grad():
        model.linear.weight.fill_(1.0)

    es = EarlyStopping(
        monitor="val_auc",
        mode="auto",
        patience=1,
        verbose=1,
        restore_best_weights=True,
    )
    es.set_model(model)
    es.on_train_begin()

    # Missing metric should be ignored.
    es.on_epoch_end(0, {})

    # Improvement stores best weights.
    es.on_epoch_end(0, {"val_auc": 0.9})
    with torch.no_grad():
        model.linear.weight.fill_(2.0)

    # No improvement triggers early stop and restores best weights.
    es.on_epoch_end(1, {"val_auc": 0.8})
    es.on_train_end()
    out = capsys.readouterr().out

    assert model.stop_training is True
    assert torch.allclose(model.linear.weight, torch.tensor([[1.0]]))
    assert "early stopping" in out


def test_model_checkpoint_paths(tmp_path, capsys):
    with pytest.raises(ValueError):
        ModelCheckpoint(filepath=str(tmp_path / "bad.ckpt"), mode="unsupported")

    model = TinyModel()

    # Auto mode with an "auc" metric goes through max branch.
    ckpt_auto = ModelCheckpoint(filepath=str(tmp_path / "auto.ckpt"), monitor="val_auc", mode="auto")
    ckpt_auto.set_model(model)

    # save_best_only + missing monitor logs
    best_path = tmp_path / "best" / "model.pt"
    ckpt_best = ModelCheckpoint(
        filepath=str(best_path),
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    ckpt_best.set_model(model)
    ckpt_best.on_epoch_end(0, {})
    ckpt_best.on_epoch_end(1, {"val_loss": 0.2})
    ckpt_best.on_epoch_end(2, {"val_loss": 0.3})
    assert best_path.exists()

    # period gate + normal save + full model save path
    regular_path = tmp_path / "regular" / "full_model.pt"
    ckpt_regular = ModelCheckpoint(
        filepath=str(regular_path),
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        period=2,
    )
    ckpt_regular.set_model(model)
    ckpt_regular.on_epoch_end(0, {"loss": 0.2})
    assert not regular_path.exists()
    ckpt_regular.on_epoch_end(1, {"loss": 0.1})
    assert regular_path.exists()

    output = capsys.readouterr().out
    assert "saving model" in output
