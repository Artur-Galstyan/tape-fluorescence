import json
import os
import pathlib
import pickle
from pathlib import Path

import coolname
import equinox as eqx
import grain.python as grain
import jax
import jax.numpy as jnp
import jax.sharding as jshard
import lmdb
import mlflow
import numpy as np
import optax
from beartype.typing import SupportsIndex
from fire import Fire
from jaxonlayers.layers import TransformerEncoder
from jaxtyping import Array, Int, PRNGKeyArray, PyTree
from pydantic import BaseModel
from scipy.stats import spearmanr
from tqdm import tqdm


def setup_mlflow(experiment_name: str = "TAPE Fluorescence Prediction"):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    assert tracking_uri is not None

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def create_lmdb_dataset(
    data: list, lmdb_path: str, map_size: int = 10 * 1024 * 1024 * 1024
):
    print(f"Creating LMDB dataset at {lmdb_path=}")
    path = Path(lmdb_path)
    path.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(path),
        map_size=map_size,
        subdir=True,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    with env.begin(write=True) as txn:
        for idx, sample in enumerate(data):
            key = str(idx).encode()
            value = pickle.dumps(sample)
            txn.put(key, value)

    env.close()
    print(f"Wrote {len(data)} samples to {lmdb_path}")


class LMDBDataSource(grain.RandomAccessDataSource):
    def __init__(self, lmdb_path: str):
        self.lmdb_path = lmdb_path
        self.env = None
        self._length = None
        self._open_env()

    def _open_env(self):
        self.env = lmdb.open(
            self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        )
        with self.env.begin() as txn:
            self._length = txn.stat()["entries"]

    def __len__(self) -> int:
        assert self._length is not None
        return self._length

    def __getitem__(self, record_key: SupportsIndex):
        if self.env is None:
            self._open_env()
        assert self.env is not None
        with self.env.begin() as txn:
            data = txn.get(str(record_key).encode())
        if data is None:
            raise KeyError(f"Index {record_key} not found in LMDB")
        return pickle.loads(data)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_env()


class TokenizeSequence(grain.MapTransform):
    def __init__(self, max_protein_length: int = 512, vocab: dict | None = None):
        self.max_protein_length = max_protein_length
        if vocab is None:
            self.vocab = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        else:
            self.vocab = vocab
        self.pad_idx = len(self.vocab)

    def map(self, element: dict) -> dict:
        seq = element["primary"]
        tokens = [self.vocab.get(aa, self.pad_idx) for aa in seq]
        tokens = tokens[: self.max_protein_length]
        padding = [self.pad_idx] * (self.max_protein_length - len(tokens))
        element["tokens"] = np.array(tokens + padding, dtype=np.int32)
        element["protein_length"] = min(
            element["protein_length"], self.max_protein_length
        )
        return element


class ExtractArrays(grain.MapTransform):
    def map(self, element: dict) -> dict:
        return {
            "tokens": element["tokens"],
            "log_fluorescence": np.array(element["log_fluorescence"], dtype=np.float32),
            "num_mutations": np.array(element["num_mutations"], dtype=np.float32),
            "protein_length": element["protein_length"],
        }


def create_dataloader(
    lmdb_path: str,
    batch_size: int,
    max_protein_length: int = 512,
    shuffle: bool = True,
    seed: int = 42,
    num_epochs: int = 1,
    num_workers: int = 4,
    vocab: dict | None = None,
):
    data_source = LMDBDataSource(lmdb_path)

    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=num_epochs,
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.NoSharding(),
    )

    transformations = [
        TokenizeSequence(max_protein_length=max_protein_length, vocab=vocab),
        ExtractArrays(),
        grain.Batch(batch_size=batch_size, drop_remainder=True),
    ]

    loader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=num_workers,
        worker_buffer_size=2,
    )

    return loader, len(data_source)


class Model(eqx.Module):
    embedding: eqx.nn.Embedding
    encoder: TransformerEncoder
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    mlp: eqx.nn.MLP
    pad_idx: int

    inference: bool

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        pad_idx: int,
        emb_size: int = 128,
        *,
        key: PRNGKeyArray,
        inference: bool = False,
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.encoder = TransformerEncoder(d_model=emb_size, n_heads=4, key=k1)
        self.pad_idx = pad_idx
        self.embedding = eqx.nn.Embedding(vocab_size, emb_size, key=k2)

        self.conv1 = eqx.nn.Conv1d(128, 256, kernel_size=5, padding=2, key=k3)
        self.conv2 = eqx.nn.Conv1d(256, 512, kernel_size=5, padding=2, key=k4)

        self.mlp = eqx.nn.MLP(512, 1, width_size=128, depth=2, key=k5)
        self.inference = inference

    def __call__(
        self, x: Int[Array, "max_protein_length"], key: PRNGKeyArray | None = None
    ):
        print("MODEL JIT")
        if not self.inference:
            assert key is not None
        mask = (x != self.pad_idx).astype(jnp.float32)[:, None]
        x = eqx.filter_vmap(self.embedding)(x)  # (L, emb_size)
        x = self.encoder(x, key=key)
        x = jnp.transpose(x)  # (128, L)

        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))

        x = jnp.max(x, axis=-1)  # (512,)

        return self.mlp(x)


@eqx.filter_jit
def loss_fn(model: PyTree, X: tuple[Array, ...], y: Array, key):
    keys = jax.random.split(key, len(y))
    preds = eqx.filter_vmap(model)(X, keys)
    return jnp.mean(optax.l2_loss(preds, y))


def evaluate(model: PyTree, loader: grain.DataLoader):
    all_preds = []
    all_targets = []
    inference_model = eqx.filter_jit(eqx.filter_vmap(eqx.nn.inference_mode(model)))
    for batch in loader:
        tokens = jnp.array(batch["tokens"])
        log_fluorescences = jnp.array(batch["log_fluorescence"])

        preds = inference_model(tokens)

        all_preds.append(np.array(preds))
        all_targets.append(np.array(log_fluorescences))

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    rho, _ = spearmanr(all_preds, all_targets)

    mse = np.mean((all_preds - all_targets) ** 2)

    return mse, rho


@eqx.filter_jit(donate="all")
def step_fn(
    model: PyTree,
    X: tuple[Array, ...],
    y: Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: PRNGKeyArray,
):
    value, grads = eqx.filter_value_and_grad(loss_fn)(model, X, y, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, value


class TrainConfig(BaseModel):
    max_protein_length: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int


def main():
    # num_devices = 1
    num_devices = len(jax.devices())
    # Use Auto axis type to maintain compatibility with eqx.filter_shard / with_sharding_constraint
    mesh = jax.make_mesh((num_devices,), ("batch",), axis_types=(jshard.AxisType.Auto,))
    data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
    model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())

    setup_mlflow()

    data_dir = "data/fluorescence"
    for json_file in os.listdir(data_dir):
        if not pathlib.Path(
            f"data/fluorescence/{json_file.replace('json', 'lmdb')}"
        ).exists():
            with open(f"{data_dir}/{json_file}", "rb") as f:
                data = json.load(f)
                create_lmdb_dataset(
                    data, f"{data_dir}/{json_file.replace('json', 'lmdb')}"
                )

    model_name = coolname.generate_slug(3)
    assert model_name is not None
    print(f"{model_name=}")

    vocab = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    train_config = TrainConfig(
        max_protein_length=256,
        batch_size=256,
        learning_rate=3e-4,
        num_epochs=30,
        warmup_steps=200,
    )

    train_loader, train_data_length = create_dataloader(
        lmdb_path="data/fluorescence/fluorescence_train.lmdb",
        batch_size=train_config.batch_size,
        max_protein_length=train_config.max_protein_length,
        shuffle=True,
        num_epochs=train_config.num_epochs,
        num_workers=24,
        vocab=vocab,
    )

    steps_per_epoch = train_data_length // train_config.batch_size

    model = Model(
        len(vocab),
        train_config.max_protein_length,
        pad_idx=len(vocab),
        key=jax.random.key(22),
    )

    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_config.learning_rate,
        warmup_steps=1 * steps_per_epoch,
        decay_steps=train_config.num_epochs * steps_per_epoch,
        end_value=train_config.learning_rate * 0.01,
    )

    optimiser = optax.adamw(learning_rate=scheduler)
    opt_state = optimiser.init(eqx.filter(model, eqx.is_array))

    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)
    key = jax.random.key(0)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(train_config.model_dump())

        for step, batch in tqdm(enumerate(train_loader)):
            tokens = jnp.array(batch["tokens"])
            num_mutations = batch["num_mutations"]
            log_fluorescences = jnp.array(batch["log_fluorescence"])

            current_lr = scheduler(step)
            mlflow.log_metric("learning_rate", current_lr.item(), step=step)

            key, subkey = jax.random.split(key)
            X, y = eqx.filter_shard((tokens, log_fluorescences), data_sharding)

            model, opt_state, loss = step_fn(model, X, y, optimiser, opt_state, subkey)
            loss_val = loss.item()
            mlflow.log_metric("train_loss", loss_val, step=step)
            if step % steps_per_epoch == 0:
                test_loader, _ = create_dataloader(
                    lmdb_path="data/fluorescence/fluorescence_test.lmdb",
                    batch_size=train_config.batch_size,
                    max_protein_length=train_config.max_protein_length,
                    shuffle=False,
                    num_epochs=1,
                    num_workers=0,
                    vocab=vocab,
                )
                valid_loader, _ = create_dataloader(
                    lmdb_path="data/fluorescence/fluorescence_valid.lmdb",
                    batch_size=train_config.batch_size,
                    max_protein_length=train_config.max_protein_length,
                    shuffle=False,
                    num_epochs=1,
                    num_workers=0,
                    vocab=vocab,
                )

                epoch = step // steps_per_epoch
                test_loss, test_spearmanr = evaluate(model, test_loader)
                valid_loss, valid_spearmanr = evaluate(model, valid_loader)
                tqdm.write(
                    f"Epoch {epoch} | Train: {loss_val:.4f} | "
                    f"Valid: {valid_loss:.4f} | Test: {test_loss:.4f} "
                    f"Valid Spearman's Rho: {valid_spearmanr:.4f} | Test Spearman's Rho: {test_spearmanr:.4f}"
                )

                mlflow.log_metric("test_loss", test_loss, step=step)
                mlflow.log_metric("valid_loss", valid_loss, step=step)
                mlflow.log_metric("test_spearmanr", test_spearmanr, step=step)
                mlflow.log_metric("valid_spearmanr", valid_spearmanr, step=step)


if __name__ == "__main__":
    main()
