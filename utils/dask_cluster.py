import dask
import dask.dataframe as dd

from dask_jobqueue import SGECluster
from dask.distributed import Client

def load_register_table(data_asset, table, **kwargs):
    return dd.read_parquet(f'/wynton/protected/project/ic/data/parquet/{data_asset}/{table}/', **kwargs)

def load_cluster(cores=4, queue="long.q", memory="64GiB", walltime='04:00:00', scale=400, **kwargs):
    """
    Wrapper for loading cluster
    >>load_cluster(cores=4, queue="long.q", memory="64GiB", walltime='04:00:00', scale=400)
    """
    i = 0
    while True:
        try:
            cluster =  SGECluster(
                queue = queue,
                cores = cores,
                memory = memory,
                walltime = '04:00:00',
                death_timeout = 60,
                local_directory = f'{os.getcwd()}/dask_temp',
                log_directory = f'{os.getcwd()}/dask_temp/dask_log',
                python = sys.executable,
                resource_spec='x86-64-v=3',
                scheduler_options = {
                    'host': ''
                }
            )
        except:
            pass
        else:
            print(f'Using Port {40000 + i}...')
            break
        i += 1
        print(i)

    cluster.scale(scale)
    client = Client(cluster)
    print(client.dashboard_link)