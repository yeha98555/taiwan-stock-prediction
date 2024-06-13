from datetime import timedelta

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="d_stock_src_to_dim",
    default_args=default_args,
    description="stock data pipeline",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["stock"],
)
def d_stock_src_to_dim():
    @task
    def et_load_stock_src_data() -> dict:
        pass

    @task
    def l_upload_to_gcs(data_dict: dict):
        pass

    @task
    def l_create_bq_tables():
        pass

    @task
    def etl_create_dim_techindex_table():
        pass

    @task
    def etl_create_fact_stockprice_table():
        pass

    @task
    def etl_create_dim_financial_table():
        pass

    @task
    def l_download_dim_techindex_to_csv():
        pass

    @task
    def l_download_fact_stockprice_to_csv():
        pass

    @task
    def l_download_dim_financial_to_csv():
        pass

    data_dict = et_load_stock_src_data()
    upload_task = l_upload_to_gcs(data_dict)
    create_bq_tables_task = l_create_bq_tables()

    dim_techindex_task = etl_create_dim_techindex_table()
    fact_stockprice_task = etl_create_fact_stockprice_table()
    dim_financial_task = etl_create_dim_financial_table()

    download_dim_techindex_task = l_download_dim_techindex_to_csv()
    download_fact_stockprice_task = l_download_fact_stockprice_to_csv()
    download_dim_financial_task = l_download_dim_financial_to_csv()

    (
        upload_task
        >> create_bq_tables_task
        >> [dim_techindex_task, fact_stockprice_task, dim_financial_task]
    )
    dim_techindex_task >> download_dim_techindex_task
    fact_stockprice_task >> download_fact_stockprice_task
    dim_financial_task >> download_dim_financial_task


d_stock_src_to_dim()
