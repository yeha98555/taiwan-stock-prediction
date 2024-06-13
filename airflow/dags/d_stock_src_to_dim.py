import json
import os
from datetime import timedelta
from io import BytesIO

import pandas as pd
from google.cloud import bigquery, storage
from utils.gcp import build_bq_from_gcs, query_bq, query_bq_to_df, upload_df_to_gcs

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

RAW_BUCKET = os.environ.get("RAW_BUCKET")
PROCESSED_BUCKET = os.environ.get("PROCESSED_BUCKET")
DATASET_PREFIX = os.environ.get("DATASET_PREFIX")
DATA_DIR = "/opt/airflow"
GCS_CLIENT = storage.Client()
BQ_CLIENT = bigquery.Client()

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
    def et_load_stock_src_data() -> json:
        bucket = GCS_CLIENT.bucket(RAW_BUCKET)

        blob_name = "output_clean_date_technical.json"
        blob = bucket.blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(
                f"file {blob_name} not found in bucket {RAW_BUCKET}"
            )

        bytes_data = blob.download_as_bytes()
        data_io = BytesIO(bytes_data)
        data = json.load(data_io)

        return data

    @task
    def t_split_data_to_tables(data: json) -> dict:
        merged_dict = {}
        for item in data:
            if item == "historicalPriceFull":
                symbol = ""
                for entry in data[item]:
                    if "symbol" in entry:
                        symbol = data[item][entry]
                    else:
                        df = pd.json_normalize(data[item][entry])
                        df["symbol"] = symbol
            else:
                df = pd.json_normalize(data[item])
            merged_dict[item] = df

        return merged_dict

    @task
    def l_upload_to_gcs(data_dict: dict) -> dict:
        blob_dict = {}
        for table_name, df in data_dict.items():
            blob_name = f"processed_{table_name}.parquet"
            blob_dict[table_name] = blob_name
            upload_df_to_gcs(GCS_CLIENT, PROCESSED_BUCKET, blob_name, df)
        return blob_dict

    @task
    def l_create_bq_tables(blob_dict: dict):
        dataset_name = f"{DATASET_PREFIX}ods"
        for table_name, blob_name in blob_dict.items():
            build_bq_from_gcs(
                BQ_CLIENT,
                dataset_name,
                table_name,
                PROCESSED_BUCKET,
                blob_name,
                partition_by="date",
            )
            print(f"table {dataset_name}.{table_name} created")

    @task
    def etl_create_dim_techindex_table():
        # select columns
        query = f"""
        CREATE OR REPLACE TABLE `{DATASET_PREFIX}dim.techIndex` AS
        SELECT
            DATE(date) as date,
            close,
            volume,
            sma,
            wma,
            rsi,
            adx,
            standardDeviation,
        FROM `{DATASET_PREFIX}ods.tech60`
        """
        query_bq(BQ_CLIENT, query)
        print("table {DATASET_PREFIX}dim.techIndex created")

    @task
    def etl_create_fact_stockprice_table():
        query = f"""
        CREATE OR REPLACE TABLE `{DATASET_PREFIX}fact.stockPrice` AS
        SELECT
            DATE(date) as date,
            symbol,
            open,
            high,
            low,
            close,
            volume,
        FROM `{DATASET_PREFIX}ods.historicalPriceFull`
        """
        query_bq(BQ_CLIENT, query)
        print("table {DATASET_PREFIX}fact.stockPrice created")

    @task
    def etl_create_dim_financial_table():
        query = f"""
        CREATE OR REPLACE TABLE `{DATASET_PREFIX}dim.financial` AS
        SELECT
            DATE(r.date) as date,
            r.symbol,
            r.cashFlowToDebtRatio as cashFlowToDebtRatio,
            cfg.growthNetCashProvidedByOperatingActivites / bsg.growthTotalCurrentLiabilities as cashFlowAdequacyRatio,
            (cfg.growthNetCashProvidedByOperatingActivites - cfg.growthDividendsPaid) / (bsg.growthTotalAssets - bsg.growthTotalDebt) as cashReinvestmentRatio, 
            bsg.growthCashAndCashEquivalents / bsg.growthTotalAssets as CashtoAssetsRatio,
            r.daysOfSalesOutstanding as daysOfSalesOutstanding,
            r.assetTurnover as assetTurnover,
            r.daysOfInventoryOutstanding as daysOfInventoryOutstanding,
            r.operatingCycle as operatingCycle,
            r.grossProfitMargin as grossProfitMargin,
            r.operatingProfitMargin as operatingProfitMargin,
            r.netProfitMargin as netProfitMargin,
            fg.epsgrowth as epsgrowth,
            r.returnOnEquity as returnOnEquity,
            r.debtRatio as debtRatio,
            r.longTermDebtToCapitalization as longTermDebtToCapitalization,
            r.currentRatio as currentRatio,
            r.quickRatio as quickRatio,
        FROM `{DATASET_PREFIX}ods.financialGrowth` fg
        JOIN `{DATASET_PREFIX}ods.ratios` r
            ON fg.date = r.date and fg.symbol = r.symbol
        JOIN `{DATASET_PREFIX}ods.cashFlowStatementGrowth` cfg
            ON fg.date = cfg.date and fg.symbol = cfg.symbol
        JOIN `{DATASET_PREFIX}ods.incomeStatementGrowth` isg
            ON fg.date = isg.date and fg.symbol = isg.symbol
        JOIN `{DATASET_PREFIX}ods.balanceSheetStatementGrowth` bsg
            ON fg.date = bsg.date and fg.symbol = bsg.symbol
        """
        query_bq(BQ_CLIENT, query)
        print("table {DATASET_PREFIX}dim.financial created")

    @task
    def l_download_dim_techindex_to_csv():
        query = f"""
        SELECT *
        FROM `{DATASET_PREFIX}dim.techIndex`
        """
        df = query_bq_to_df(BQ_CLIENT, query)
        df.to_csv(f"{DATA_DIR}/dim_techindex.csv", index=False)

    @task
    def l_download_fact_stockprice_to_csv():
        query = f"""
        SELECT *
        FROM `{DATASET_PREFIX}fact.stockPrice`
        """
        df = query_bq_to_df(BQ_CLIENT, query)
        df.to_csv(f"{DATA_DIR}/fact_stockprice.csv", index=False)

    @task
    def l_download_dim_financial_to_csv():
        query = f"""
        SELECT *
        FROM `{DATASET_PREFIX}dim.financial`
        """
        df = query_bq_to_df(BQ_CLIENT, query)
        df.to_csv(f"{DATA_DIR}/dim_financial.csv", index=False)

    data = et_load_stock_src_data()
    data_dict = t_split_data_to_tables(data)
    blob_dict = l_upload_to_gcs(data_dict)
    create_bq_tables_task = l_create_bq_tables(blob_dict)

    dim_techindex_task = etl_create_dim_techindex_table()
    fact_stockprice_task = etl_create_fact_stockprice_table()
    dim_financial_task = etl_create_dim_financial_table()

    download_dim_techindex_task = l_download_dim_techindex_to_csv()
    download_fact_stockprice_task = l_download_fact_stockprice_to_csv()
    download_dim_financial_task = l_download_dim_financial_to_csv()

    (
        create_bq_tables_task
        >> [dim_techindex_task, fact_stockprice_task, dim_financial_task]
    )
    dim_techindex_task >> download_dim_techindex_task
    fact_stockprice_task >> download_fact_stockprice_task
    dim_financial_task >> download_dim_financial_task


d_stock_src_to_dim()
