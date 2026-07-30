import os

from ..constants import DF_TYPES, DIR, DOC, OPERA_RAW_DATA_DIR, OPERA_WORK_ROOT, QUERY, SPLITS


def get_raw_data(dataset_name, df_type):
    assert df_type in DF_TYPES, f"Not supported dataframe type: {df_type}. All supported types are: {DF_TYPES}."

    data_dir = os.path.join(OPERA_RAW_DATA_DIR, dataset_name.replace("/", "_"))
    os.makedirs(data_dir, exist_ok=True)

    if df_type == DIR:
        return data_dir
    else:
        return os.path.join(data_dir, f"{df_type}.parquet")


def get_save_path(save_name):
    save_path = os.path.join(OPERA_WORK_ROOT, save_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def get_data(save_path, df_type, split=None, fallback=False, dataset_name=None):
    assert df_type in DF_TYPES, f"Not supported dataframe type: {df_type}. All supported types are: {DF_TYPES}."
    assert split in SPLITS, f"Not supported split type: {split}. All supported types are: {SPLITS}."

    data_dir = os.path.join(save_path, split)

    if df_type == DIR:
        return data_dir
    else:
        os.makedirs(data_dir, exist_ok=True)
        dataframe_path = os.path.join(data_dir, f"{df_type}.parquet")

    if not os.path.exists(dataframe_path) and fallback:
        return get_raw_data(dataset_name=dataset_name, df_type=df_type)
    else:
        return dataframe_path


def get_model_save_path(config, return_path=True):
    save_path = config.save_path
    assert save_path, f"Invalid save_path: {save_path}"

    model_save_name = f"model_ft-{config.optimization.learning_rate}_{config.optimization.temperature}T_{config.optimization.weight_decay}wd"
    model_save_name += f"-mine_{config.mining.mode}"
    model_save_name += f"-opera{config.opera.level}"

    if config.optimization.max_steps > 0:
        model_save_name += f"-{config.optimization.max_steps}it"
    else:
        assert config.optimization.num_train_epochs > 0
        model_save_name += f"-{config.optimization.num_train_epochs}e"

    if return_path:
        return os.path.join(save_path, model_save_name)
    else:
        return model_save_name


def get_result_path(encoder_name, config):
    if os.path.exists(encoder_name):
        return os.path.join(encoder_name, "results.json")
    else:
        return os.path.join(config.save_path, f"{encoder_name.replace('/', '_')}-results.json")


def get_emb_cache_files(dataset_name, model_name, data_type):
    assert data_type in [DOC, QUERY]
    # cache_location/dataset_name/model_name/QUERY_emb.npy
    # cache_location/dataset_name/model_name/QUERY_id.npy
    emb_root = os.path.join(OPERA_WORK_ROOT, dataset_name.replace("/", "_"), model_name.replace("/", "_"))
    os.makedirs(emb_root, exist_ok=True)
    emb_file = os.path.join(emb_root, f"{data_type}_emb.npy")
    id_file = os.path.join(emb_root, f"{data_type}_id.npy")
    return emb_file, id_file
